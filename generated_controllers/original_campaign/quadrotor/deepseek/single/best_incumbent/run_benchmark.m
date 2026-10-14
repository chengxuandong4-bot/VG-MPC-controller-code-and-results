function run_benchmark(bm, mode)
% RUN_BENCHMARK  Run benchmark with full first-round-compatible output.
    addpath(genpath(fileparts(mfilename('fullpath'))));
    rng(42);

    switch bm
        case 'cstr',   c = config_lightweight(mode);
        case 'auv',    c = config_fault_tolerant(mode);
        case 'rampc',  c = config_adaptive(mode);
        otherwise, error('Unknown: %s', bm);
    end

    p = c.init_params();
    r = c.run_baseline(p);
    m0 = c.compute_metrics(r, p);
    q0 = c.aggregate_score(m0);
    rd = c.results_dir;
    if ~exist(rd,'dir'), mkdir(rd); end

    % --- Run ---
    if strcmp(mode, 'single')
        all_log = run_single(c, p, r, q0, m0, rd);
    else
        all_log = run_dual(c, p, r, rd);
    end

    % --- Generate final outputs ---
    generate_final_outputs(rd, all_log, bm, mode, q0);
    fprintf('=== %s %s DONE ===\n', bm, mode);
    exit;
end


function log = run_single(c, p, r, q0, m0, rd)
    inc = struct('c',r,'q',q0,'g',0);
    log = repmat(struct('gen',NaN,'name','','q',NaN,'acc',false,'upd',false), c.budget_T, 1);
    fprintf('[%s single] baseline q=%.4f T=%d\n', c.name, q0, c.budget_T);

    for t = 1:c.budget_T
        if t == 1, cand = c.single.initial_candidate(p, r);
        else, cand = c.single.generate_candidate(inc, struct('text',''), p, t); end

        [qs, ss, vv, met, res, acc] = evaluate(c, p, cand);
        upd = false;
        if acc && qs > inc.q, inc.c = cand; inc.q = qs; inc.g = t; upd = true; end

        if upd || mod(t,50)==0 || t==c.budget_T
            fprintf('[%03d/%d] %s q=%.4f\n', t, c.budget_T, ternary(upd,'NEW BEST',''), qs);
        end

        save_gen_snapshot(rd, t, cand, met, qs, ss, vv, acc, upd, 'S', m0);
        log(t) = struct('gen',t,'name',cand.name,'q',qs,'acc',acc,'upd',upd);
    end
    fprintf('=== Best: gen %d q=%.4f ===\n', inc.g, inc.q);
end


function log = run_dual(c, p, r, rd)
    inc = struct('c',[],'q',-inf,'g',0);
    sA=0; sB=0; gA=c.dual.generator_A; gB=c.dual.generator_B;
    log = repmat(struct('gen',NaN,'name','','q',NaN,'acc',false,'upd',false,'src',''), c.budget_T*2, 1);
    li = 0;
    fprintf('[%s %s] T=%d\n', c.name, c.mode, c.budget_T);

    for t = 1:c.budget_T
        cA = ternary(t==1, gA.initial_candidate(p,r), gA.generate_candidate(inc,struct('text',''),p,t,'A'));
        cB = ternary(t==1, gB.initial_candidate(p,r), gB.generate_candidate(inc,struct('text',''),p,t,'B'));

        [qA,sA_s,vA,mA,rA,accA] = evaluate(c, p, cA);
        [qB,sB_s,vB,mB,rB,accB] = evaluate(c, p, cB);

        dq = qA - qB;
        if ~accA&&accB, w='B'; elseif accA&&~accB, w='A';
        elseif ~accA&&~accB, w='N'; elseif dq>c.epsilon, w='A';
        elseif dq<-c.epsilon, w='B'; else, w='T'; end

        upd = false;
        if strcmp(w,'A')&&qA>inc.q, inc.c=cA; inc.q=qA; inc.g=t; upd=true; sA=0; sB=sB+1;
        elseif strcmp(w,'B')&&qB>inc.q, inc.c=cB; inc.q=qB; inc.g=t; upd=true; sB=0; sA=sA+1;
        else, sA=sA+1; sB=sB+1; end

        if mod(t,20)==0||t==1, fprintf('[%03d/%d] A:q=%.4f B:q=%.4f win=%s best=%.4f\n',t,c.budget_T,qA,qB,w,inc.q); end

        save_gen_snapshot(rd, t, cA, mA, qA, sA_s, vA, accA, strcmp(w,'A'), 'A');
        save_gen_snapshot(rd, t, cB, mB, qB, sB_s, vB, accB, strcmp(w,'B'), 'B');

        li=li+1; log(li)=struct('gen',t,'name',cA.name,'q',qA,'acc',accA,'upd',strcmp(w,'A')&&upd,'src','A');
        li=li+1; log(li)=struct('gen',t,'name',cB.name,'q',qB,'acc',accB,'upd',strcmp(w,'B')&&upd,'src','B');
    end
    log = log(1:li);
    fprintf('=== Best: gen %d q=%.4f ===\n', inc.g, inc.q);
end


function [q, s, v, met, res, acc] = evaluate(c, p, cand)
    res = c.simulate_closed_loop(p, cand);
    met = c.compute_metrics(res, p);
    s = c.aggregate_score(met);
    v = c.compute_violations(res, p);
    q = s - c.lambda_v * v;
    acc = c.acceptance_gate(met, res, p);
end


function generate_final_outputs(rd, log, bm, mode, baseline_q)
    log = log(~isnan([log.gen]));
    n = length(log);

    % Find best
    valid = log([log.acc]);
    if isempty(valid), [~,bi]=max([log.q]); best=log(bi);
    else, [~,bi]=max([valid.q]); best=valid(bi); end

    % 1. summary_metrics.json
    sum_struct.mode = mode; sum_struct.benchmark = bm;
    sum_struct.T_max = max([log.gen]); sum_struct.T_actual = n;
    sum_struct.best_gen = best.gen; sum_struct.best_q = best.q;
    sum_struct.best_name = best.name; sum_struct.baseline_q = baseline_q;
    sum_struct.n_accepted = sum([log.acc]); sum_struct.n_updated = sum([log.upd]);
    sum_struct.timestamp = datestr(now);
    write_json(fullfile(rd,'summary_metrics.json'), sum_struct);

    % 2. 指标分析报告.txt
    fid = fopen(fullfile(rd,'指标分析报告.txt'),'w','n','UTF-8');
    fprintf(fid,'%s %s — 指标分析报告\n', bm, mode);
    fprintf(fid,'生成时间: %s\n', datestr(now));
    fprintf(fid,'总代数: %d, 有效候选: %d, 更新次数: %d\n', n, sum_struct.n_accepted, sum_struct.n_updated);
    fprintf(fid,'基线 q=%.4f, 最佳 q=%.4f (gen %d, +%.1f%%)\n', baseline_q, best.q, best.gen, (best.q/baseline_q-1)*100);
    fclose(fid);

    % 3. campaign_summary.csv
    fid = fopen(fullfile(rd,'campaign_summary.csv'),'w','n','UTF-8');
    fprintf(fid,'gen,name,q_score,accepted,updated\n');
    for i=1:n, fprintf(fid,'%d,%s,%.6f,%d,%d\n',log(i).gen,log(i).name,log(i).q,log(i).acc,log(i).upd); end
    fclose(fid);

    % 4. candidate_leaderboard.csv
    [~,si] = sort([log.q],'descend');
    fid = fopen(fullfile(rd,'candidate_leaderboard.csv'),'w','n','UTF-8');
    fprintf(fid,'rank,gen,name,q_score,accepted,updated\n');
    for i=1:n
        j=si(i);
        fprintf(fid,'%d,%d,%s,%.6f,%d,%d\n',i,log(j).gen,log(j).name,log(j).q,log(j).acc,log(j).upd);
    end
    fclose(fid);

    % 5. generation_manifest.csv
    fid = fopen(fullfile(rd,'generation_manifest.csv'),'w','n','UTF-8');
    fprintf(fid,'generation_index,name,q_score,accepted,updated,snapshot_dir\n');
    for i=1:n
        fprintf(fid,'%d,%s,%.6f,%d,%d,generation_snapshots/gen_%03d\n',log(i).gen,log(i).name,log(i).q,log(i).acc,log(i).upd,log(i).gen);
    end
    fclose(fid);

    % 6. baseline_vs_best.png placeholder
    try
        fig=figure('Visible','off'); plot(1,1); title(sprintf('%s %s: best q=%.4f',bm,mode,best.q));
        saveas(fig,fullfile(rd,'baseline_vs_best.png')); close(fig);
    catch
    end

    fprintf('Final outputs written to %s\n', rd);
end


function write_json(path, data)
    fid = fopen(path,'w','n','UTF-8');
    if fid<1, return; end
    try, fprintf(fid,'%s',jsonencode(data)); catch, fprintf(fid,'{}'); end
    fclose(fid);
end


function v = ternary(cond, tval, fval)
    if cond, v = tval; else, v = fval; end
end
