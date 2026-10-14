function run_unified_outer_loop(benchmark_config)
    root_dir = benchmark_config.root_dir;
    addpath(genpath(fullfile(root_dir, 'src')));
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')))));

    fprintf('\n=== Baseline ===\n');
    params = benchmark_config.init_params();
    base_result = benchmark_config.run_baseline(params);

    switch benchmark_config.mode
        case 'single'
            run_single_loop(benchmark_config, params, base_result);
        case {'dual_obj', 'dual_strategy'}
            run_dual_loop(benchmark_config, params, base_result);
    end
end


function run_single_loop(cfg, params, base_result)
    T = cfg.budget_T; rd = cfg.results_dir;
    if ~exist(rd,'dir'), mkdir(rd); end
    sd = fullfile(rd,'gen_snapshots');
    if ~exist(sd,'dir'), mkdir(sd); end

    inc = struct('c',[],'q',-inf,'m',[],'g',0);
    stall = 0; fb = struct('text','');

    for t = 1:T
        if t == 1
            cand = cfg.single.initial_candidate(params, base_result);
        elseif stall > 0 && mod(stall, cfg.restart_interval) == 0
            cand = cfg.single.restart_mutate(inc, params, stall);
        else
            cand = cfg.single.generate_candidate(inc, fb, params, t);
        end

        [ok, diag] = cfg.parse_and_validate(cand);
        if ~ok
            fb = cfg.build_error_feedback(diag); stall = stall+1;
            if stall >= cfg.early_stop_N, fprintf('Stop gen %d stall=%d\n', t, stall); break; end
            continue;
        end

        res = cfg.simulate_closed_loop(params, cand);
        m = cfg.compute_metrics(res, params);
        s = cfg.aggregate_score(m);
        v = cfg.compute_violations(res, params);
        q = s - cfg.lambda_v * v;
        acc = cfg.acceptance_gate(m, res, params);

        updated = false;
        if acc && q > inc.q
            inc.c = cand; inc.q = q; inc.m = m; inc.g = t; updated = true; stall = 0;
            fprintf('[%03d/%d] NEW BEST q=%.4f %s\n', t, T, q, cand.name);
        else
            stall = stall + 1;
            flag = ''; if ~acc, flag = '[REJ]'; end
            fprintf('[%03d/%d] q=%.4f %s %s\n', t, T, q, cand.name, flag);
        end

        fb = cfg.build_feedback(inc, m, s, q, diag, updated);

        % Save snapshot
        try
            gd = fullfile(sd, sprintf('g%03d_%s', t, matlab.lang.makeValidName(cand.name)));
            if ~exist(gd,'dir'), mkdir(gd); end
            save(fullfile(gd,'m.mat'),'m','s','q','acc','updated');
            fid=fopen(fullfile(gd,'fb.txt'),'w'); fprintf(fid,'%s',fb.text); fclose(fid);
        catch
        end

        if stall >= cfg.early_stop_N, fprintf('Stop gen %d stall=%d\n', t, stall); break; end
    end

    fprintf('\n=== Best: gen %d q=%.4f ===\n', inc.g, inc.q);
    try cfg.generate_report(inc); catch, end
end


function run_dual_loop(cfg, params, base_result)
    T = cfg.budget_T; rd = cfg.results_dir;
    if ~exist(rd,'dir'), mkdir(rd); end
    sd = fullfile(rd,'gen_snapshots');
    if ~exist(sd,'dir'), mkdir(sd); end

    inc = struct('c',[],'q',-inf,'m',[],'g',0);
    sA=0; sB=0; fbA=struct('text',''); fbB=struct('text','');
    gA = cfg.dual.generator_A; gB = cfg.dual.generator_B;

    for t = 1:T
        if t==1, cA=gA.initial_candidate(params,base_result); cB=gB.initial_candidate(params,base_result);
        else, cA=gA.generate_candidate(inc,fbA,params,t,'A'); cB=gB.generate_candidate(inc,fbB,params,t,'B'); end

        [okA,dA]=cfg.parse_and_validate(cA); [okB,dB]=cfg.parse_and_validate(cB);
        qA=-inf; qB=-inf; accA=false; accB=false; mA=[]; mB=[];

        if okA, rA=cfg.simulate_closed_loop(params,cA); mA=cfg.compute_metrics(rA,params);
            sA=cfg.aggregate_score(mA); vA=cfg.compute_violations(rA,params);
            qA=sA-cfg.lambda_v*vA; accA=cfg.acceptance_gate(mA,rA,params); end
        if okB, rB=cfg.simulate_closed_loop(params,cB); mB=cfg.compute_metrics(rB,params);
            sB=cfg.aggregate_score(mB); vB=cfg.compute_violations(rB,params);
            qB=sB-cfg.lambda_v*vB; accB=cfg.acceptance_gate(mB,rB,params); end

        % Adversarial compare
        dq = qA - qB;
        if ~accA&&accB, w='B'; l='A';
        elseif accA&&~accB, w='A'; l='B';
        elseif ~accA&&~accB, w='N'; l='X';
        elseif dq > cfg.epsilon, w='A'; l='B';
        elseif dq < -cfg.epsilon, w='B'; l='A';
        else, w='T'; l='T'; end

        updated=false;
        if strcmp(w,'A')&&qA>inc.q, inc.c=cA;inc.q=qA;inc.m=mA;inc.g=t;updated=true;sA=0;sB=sB+1;
        elseif strcmp(w,'B')&&qB>inc.q, inc.c=cB;inc.q=qB;inc.m=mB;inc.g=t;updated=true;sB=0;sA=sA+1;
        else, sA=sA+1;sB=sB+1; end

        fbA=cfg.build_feedback(inc,mA,sA,qA,dA,updated,w,l,dq,'A');
        fbB=cfg.build_feedback(inc,mB,sB,qB,dB,updated,w,l,dq,'B');

        fprintf('[%03d/%d] A:q=%.4f B:q=%.4f win=%s inc_q=%.4f\n',t,T,qA,qB,w,inc.q);

        try, save_dual_snap(sd,t,'A',cA,mA,qA,accA); save_dual_snap(sd,t,'B',cB,mB,qB,accB); catch, end

        if sA>=cfg.early_stop_N&&sB>=cfg.early_stop_N, fprintf('Stop gen %d\n',t); break; end
    end
    fprintf('\n=== Best: gen %d q=%.4f ===\n',inc.g,inc.q);
    try cfg.generate_report(inc); catch, end
end

function save_dual_snap(sd, t, label, cand, met, q, acc)
    try
        gd = fullfile(sd, sprintf('g%03d_%s_%s', t, label, matlab.lang.makeValidName(cand.name)));
        if ~exist(gd,'dir'), mkdir(gd); end
        m = met; save(fullfile(gd,'m.mat'),'m','q','acc');
    catch
    end
end
