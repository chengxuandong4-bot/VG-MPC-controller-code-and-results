function run_cstr_repeat(rep_index, mode)
%RUN_CSTR_REPEAT Execute one frozen CSTR repetition mode.
%
% Examples:
%   run_cstr_repeat(1, 'single')
%   run_cstr_repeat(1, 'dual_objective')
%   run_cstr_repeat(1, 'dual_strategy')

    arguments
        rep_index (1,1) double {mustBeInteger, mustBePositive, mustBeLessThanOrEqual(rep_index,5)}
        mode (1,:) char {mustBeMember(mode, {'single','dual_objective','dual_strategy'})}
    end

    protocol_dir = fileparts(mfilename('fullpath'));
    repeat_root = fileparts(protocol_dir);
    deepseek_root = fileparts(repeat_root);
    rep_dir = fullfile(repeat_root, sprintf('rep_%02d', rep_index));
    results_dir = fullfile(rep_dir, mode);

    if exist(results_dir, 'dir')
        existing = dir(results_dir);
        existing = existing(~ismember({existing.name}, {'.','..'}));
        if ~isempty(existing)
            error('VGRepeat:OutputExists', ...
                'Refusing to overwrite non-empty result directory: %s', results_dir);
        end
    else
        mkdir(results_dir);
    end

    diary(fullfile(results_dir, 'run.log'));
    diary on;
    diary_cleanup = onCleanup(@() diary('off'));
    fprintf('=== CSTR repeat rep_%02d mode=%s ===\n', rep_index, mode);
    fprintf('Started: %s\n', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));

    addpath(fullfile(deepseek_root, 'unified_loop'));
    addpath(fullfile(deepseek_root, 'benchmark_lightweight'));
    addpath(fullfile(deepseek_root, 'benchmark_lightweight', 'src'));
    clear llm_wrapper;
    rng(42, 'twister');

    mode_for_config = mode;
    if strcmp(mode, 'dual_objective'), mode_for_config = 'dual_obj'; end
    c = config_lightweight(mode_for_config);
    p = c.init_params();
    baseline_result = c.run_baseline(p);
    baseline_metrics = c.compute_metrics(baseline_result, p);
    baseline_q = c.aggregate_score(baseline_metrics);
    baseline_struct = metric_struct(c.metric_names, baseline_metrics);
    fprintf('Baseline q=%.8f\n', baseline_q);

    wrapper_options = struct( ...
        'py_script_name', fullfile('..', 'cstr_repeats_v4flash', ...
            'protocol', 'llm_generate_repeat.py'), ...
        'intervene_at', 999);
    counters = struct('api',0,'fallback',0,'script_failed',0,'other',0, ...
        'accepted',0,'updated',0);
    records = empty_records();
    run_timer = tic;

    switch mode
        case 'single'
            [inc, counters, records] = run_single(c, p, baseline_metrics, ...
                baseline_struct, baseline_q, results_dir, wrapper_options, ...
                counters, records, run_timer);
            seed_q = NaN;
            seed_origin_q = NaN;

        case 'dual_objective'
            [inc, counters, records] = run_dual_objective(c, p, ...
                baseline_metrics, baseline_struct, baseline_q, results_dir, ...
                wrapper_options, counters, records, run_timer);
            seed_q = NaN;
            seed_origin_q = NaN;

        case 'dual_strategy'
            dual_dir = fullfile(rep_dir, 'dual_objective');
            [seed_cfg, seed_origin_q] = best_snapshot_candidate(dual_dir);
            [inc, seed_q, counters, records] = run_dual_strategy(c, p, ...
                baseline_metrics, baseline_struct, baseline_q, seed_cfg, ...
                results_dir, wrapper_options, counters, records, run_timer);
    end

    elapsed_s = toc(run_timer);
    total_candidates = numel(records);
    assert(total_candidates == 200, ...
        'Expected 200 candidate records, found %d', total_candidates);

    summary = struct();
    summary.status = 'completed';
    summary.replication = rep_index;
    summary.benchmark = 'cstr';
    summary.mode = mode;
    summary.model = 'deepseek-v4-flash';
    summary.thinking = 'disabled';
    summary.temperature = 0.3;
    summary.max_tokens = 512;
    summary.matlab_rng_seed = 42;
    summary.total_candidates = total_candidates;
    summary.baseline_q = baseline_q;
    summary.best_q = inc.q;
    summary.best_generation = inc.g;
    summary.gain_pct = (inc.q - baseline_q) / abs(baseline_q) * 100;
    summary.accepted_count = counters.accepted;
    summary.acceptance_rate = counters.accepted / total_candidates;
    summary.updated_count = counters.updated;
    summary.api_success_count = counters.api;
    summary.api_success_rate = counters.api / total_candidates;
    summary.fallback_count = counters.fallback;
    summary.fallback_rate = counters.fallback / total_candidates;
    summary.script_failed_count = counters.script_failed;
    summary.other_source_count = counters.other;
    summary.elapsed_s = elapsed_s;
    if strcmp(mode, 'dual_strategy')
        summary.seed_origin_q = seed_origin_q;
        summary.seed_q_recomputed = seed_q;
        summary.seed_increment_pct = (inc.q - seed_q) / abs(seed_q) * 100;
    end

    write_json(fullfile(results_dir, 'summary_metrics.json'), summary);
    write_json(fullfile(results_dir, 'best_config.json'), inc.c);
    writetable(struct2table(records), fullfile(results_dir, 'trajectory.csv'));
    fprintf('COMPLETED best_gen=%d best_q=%.8f gain=%+.3f%% elapsed=%.1fs\n', ...
        inc.g, inc.q, summary.gain_pct, elapsed_s);
    clear diary_cleanup;
    diary off;
end


function [inc, counters, records] = run_single(c, p, m0, bl, q0, rd, opts, counters, records, timer)
    base_cfg = c.single.initial_candidate(p, []);
    inc = struct('c',base_cfg,'q',q0,'g',0,'m',m0);
    stall = 0;
    for t = 1:200
        fb = struct();
        fb.text = sprintf('CSTR gen %d. Best q=%.4f at gen %d.', t, inc.q, inc.g);
        fb.baseline_metrics = bl;
        fb.stall_count = stall;
        [cand, ~, diag] = llm_wrapper(inc, fb, p, t, 'S', 'cstr', opts);
        [q, m, s, v, accepted] = evaluate_candidate(c, p, cand);
        updated = accepted && q > inc.q;
        if updated
            inc = struct('c',cand,'q',q,'g',t,'m',m);
            stall = 0;
        else
            stall = stall + 1;
        end
        counters = update_counters(counters, diag.source, accepted, updated);
        records(end+1) = make_record(t, t, 'S', q, accepted, updated, inc.q, diag.source); %#ok<AGROW>
        save_gen_snapshot(rd, t, cand, m, q, s, v, accepted, updated, 'S', m0);
        if updated || mod(t,25)==0
            fprintf('[%03d/200] q=%.6f acc=%d best=%.6f source=%s elapsed=%.0fs\n', ...
                t, q, accepted, inc.q, diag.source, toc(timer));
        end
    end
end


function [inc, counters, records] = run_dual_objective(c, p, m0, bl, q0, rd, opts, counters, records, timer)
    base_cfg = c.single.initial_candidate(p, []);
    inc = struct('c',base_cfg,'q',q0,'g',0,'m',m0);
    stall = 0;
    for t = 1:100
        fbA = struct('text',sprintf('CSTR gen %d. Role A(perf). Best q=%.4f at gen %d.',t,inc.q,inc.g), ...
            'baseline_metrics',bl,'stall_count',stall);
        fbB = struct('text',sprintf('CSTR gen %d. Role B(rob). Best q=%.4f at gen %d.',t,inc.q,inc.g), ...
            'baseline_metrics',bl,'stall_count',stall);
        [candA, ~, diagA] = llm_wrapper(inc, fbA, p, t, 'A', 'cstr', opts);
        [candB, ~, diagB] = llm_wrapper(inc, fbB, p, t, 'B', 'cstr', opts);
        [qA,mA,sA,vA,accA] = evaluate_candidate(c,p,candA);
        [qB,mB,sB,vB,accB] = evaluate_candidate(c,p,candB);
        winner = pair_winner(qA,qB,accA,accB);
        updA = false; updB = false;
        if strcmp(winner,'A') && qA > inc.q
            inc = struct('c',candA,'q',qA,'g',t,'m',mA); stall=0; updA=true;
        elseif strcmp(winner,'B') && qB > inc.q
            inc = struct('c',candB,'q',qB,'g',t,'m',mB); stall=0; updB=true;
        else
            stall = stall + 1;
        end
        counters = update_counters(counters,diagA.source,accA,updA);
        counters = update_counters(counters,diagB.source,accB,updB);
        records(end+1) = make_record(t,2*t-1,'A',qA,accA,updA,inc.q,diagA.source); %#ok<AGROW>
        records(end+1) = make_record(t,2*t,'B',qB,accB,updB,inc.q,diagB.source); %#ok<AGROW>
        save_gen_snapshot(rd,t,candA,mA,qA,sA,vA,accA,updA,'A',m0);
        save_gen_snapshot(rd,t,candB,mB,qB,sB,vB,accB,updB,'B',m0);
        if updA || updB || mod(t,20)==0
            fprintf('[%03d/100] A=%.6f B=%.6f winner=%s best=%.6f elapsed=%.0fs\n', ...
                t,qA,qB,winner,inc.q,toc(timer));
        end
    end
end


function [inc, seed_q, counters, records] = run_dual_strategy(c, p, m0, bl, q0, seed_cfg, rd, opts, counters, records, timer)
    seed_result = c.simulate_closed_loop(p, seed_cfg);
    seed_metrics = c.compute_metrics(seed_result, p);
    seed_q = c.aggregate_score(seed_metrics) - c.lambda_v*c.compute_violations(seed_result,p);
    inc = struct('c',seed_cfg,'q',seed_q,'g',0,'m',seed_metrics);
    fprintf('Warm-start seed recomputed q=%.8f (baseline=%.8f)\n',seed_q,q0);
    stall = 0;
    promptA = ['CSTR dual_strategy gen %d. Role A (LOCAL EXPLOIT): Precision tuner. ', ...
        'Make SMALL adjustments (3-15%%) to Wy_base, Wu, hessian_reg, obs_alpha. ', ...
        'Keep structural flags (use_adaptive_horizon, use_disturbance_observer) fixed. ', ...
        'Goal: squeeze marginal gains from the current best. ', ...
        'Best q=%.4f IAE=%.1f. Improve by 1-3%%.'];
    promptB = ['CSTR dual_strategy gen %d. Role B (STRUCTURAL EXPLORE): Explorer. ', ...
        'Try DIFFERENT architectures: toggle use_adaptive_horizon, use_disturbance_observer, ', ...
        'use_param_adaptation. Make LARGE changes (30-80%%) to parameters. ', ...
        'Try horizon_max at extremes (10 vs 30), Wy_base at extremes (0.5 vs 8), ', ...
        'hessian_reg at extremes (1e-8 vs 1e-3). ', ...
        'Goal: discover new regions. q does NOT need to beat the incumbent. ', ...
        'Current best q=%.4f. Explore freely!'];
    for t = 1:100
        fbA = struct('prompt_override',sprintf(promptA,t,inc.q,inc.m(1)), ...
            'baseline_metrics',bl,'stall_count',stall);
        fbB = struct('prompt_override',sprintf(promptB,t,inc.q), ...
            'baseline_metrics',bl,'stall_count',stall);
        [candA, ~, diagA] = llm_wrapper(inc,fbA,p,t,'A','cstr',opts);
        [candB, ~, diagB] = llm_wrapper(inc,fbB,p,t,'B','cstr',opts);
        [qA,mA,sA,vA,accA] = evaluate_candidate(c,p,candA);
        [qB,mB,sB,vB,accB] = evaluate_candidate(c,p,candB);
        winner = pair_winner(qA,qB,accA,accB);
        updA=false; updB=false;
        if strcmp(winner,'A') && qA>inc.q
            inc=struct('c',candA,'q',qA,'g',t,'m',mA); stall=0; updA=true;
        elseif strcmp(winner,'B') && qB>inc.q
            inc=struct('c',candB,'q',qB,'g',t,'m',mB); stall=0; updB=true;
        else
            stall=stall+1;
        end
        counters=update_counters(counters,diagA.source,accA,updA);
        counters=update_counters(counters,diagB.source,accB,updB);
        records(end+1)=make_record(t,2*t-1,'A',qA,accA,updA,inc.q,diagA.source); %#ok<AGROW>
        records(end+1)=make_record(t,2*t,'B',qB,accB,updB,inc.q,diagB.source); %#ok<AGROW>
        save_gen_snapshot(rd,t,candA,mA,qA,sA,vA,accA,updA,'A',m0);
        save_gen_snapshot(rd,t,candB,mB,qB,sB,vB,accB,updB,'B',m0);
        if updA || updB || mod(t,20)==0
            fprintf('[%03d/100] A=%.6f B=%.6f winner=%s best=%.6f elapsed=%.0fs\n', ...
                t,qA,qB,winner,inc.q,toc(timer));
        end
    end
end


function [q,m,s,v,accepted] = evaluate_candidate(c,p,candidate)
    result = c.simulate_closed_loop(p,candidate);
    m = c.compute_metrics(result,p);
    s = c.aggregate_score(m);
    v = c.compute_violations(result,p);
    q = s - c.lambda_v*v;
    accepted = c.acceptance_gate(m,result,p);
end


function winner = pair_winner(qA,qB,accA,accB)
    winner='T';
    if ~accA && accB, winner='B';
    elseif accA && ~accB, winner='A';
    elseif ~accA && ~accB, winner='N';
    elseif qA-qB>0.005, winner='A';
    elseif qA-qB<-0.005, winner='B';
    end
end


function [candidate,best_q] = best_snapshot_candidate(dual_dir)
    snapshot_root = fullfile(dual_dir,'generation_snapshots');
    assert(exist(snapshot_root,'dir')==7, ...
        'Corresponding dual-objective snapshots are missing: %s',snapshot_root);
    listing = dir(snapshot_root);
    best_q=-inf; candidate=[];
    for i=1:numel(listing)
        if ~listing(i).isdir || ismember(listing(i).name,{'.','..'}), continue; end
        meta_path=fullfile(listing(i).folder,listing(i).name,'meta_snapshot.json');
        cfg_path=fullfile(listing(i).folder,listing(i).name,'config_snapshot.json');
        if exist(meta_path,'file')==2 && exist(cfg_path,'file')==2
            meta=jsondecode(fileread(meta_path));
            if meta.q_score>best_q
                best_q=meta.q_score;
                candidate=jsondecode(fileread(cfg_path));
            end
        end
    end
    assert(~isempty(candidate),'No dual-objective seed candidate found');
end


function counters = update_counters(counters,source,accepted,updated)
    if strcmp(source,'llm_api'), counters.api=counters.api+1;
    elseif strcmp(source,'llm_fallback'), counters.fallback=counters.fallback+1;
    elseif strcmp(source,'script_failed'), counters.script_failed=counters.script_failed+1;
    else, counters.other=counters.other+1;
    end
    counters.accepted=counters.accepted+double(accepted);
    counters.updated=counters.updated+double(updated);
end


function rows = empty_records()
    rows = struct('generation',{},'candidate_index',{},'role',{},'q_score',{}, ...
        'accepted',{},'updated',{},'incumbent_q',{},'source',{});
end


function row = make_record(gen,index,role,q,accepted,updated,inc_q,source)
    row=struct('generation',gen,'candidate_index',index,'role',role, ...
        'q_score',q,'accepted',logical(accepted),'updated',logical(updated), ...
        'incumbent_q',inc_q,'source',source);
end


function out = metric_struct(names,values)
    out=struct();
    for i=1:min(numel(names),numel(values)), out.(names{i})=values(i); end
end


function write_json(path,data)
    fid=fopen(path,'w','n','UTF-8');
    assert(fid>0,'Cannot write JSON file: %s',path);
    cleanup=onCleanup(@() fclose(fid));
    fprintf(fid,'%s',jsonencode(data,'PrettyPrint',true));
    clear cleanup;
end
