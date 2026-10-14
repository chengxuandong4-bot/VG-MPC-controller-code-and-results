function out = evaluate_cstr_candidate(c, p, candidate)
% Evaluate through the existing CSTR simulator, metrics, score, violations, and gate.
    timer = tic;
    out = struct('candidate_id','','candidate',candidate,'execution_success',false, ...
        'exception_message','','metrics',[],'raw_score',-inf,'violation_penalty',100, ...
        'selection_score',-inf,'accepted',false,'duration_s',NaN);
    try
        result = c.simulate_closed_loop(p, candidate);
        out.execution_success = ~(isfield(result,'failed') && result.failed);
        out.metrics = c.compute_metrics(result, p);
        out.raw_score = c.aggregate_score(out.metrics);
        out.violation_penalty = c.compute_violations(result, p);
        out.selection_score = out.raw_score - c.lambda_v * out.violation_penalty;
        out.accepted = out.execution_success && c.acceptance_gate(out.metrics, result, p);
    catch ME
        out.exception_message = getReport(ME, 'basic');
        failed = struct('failed',true);
        out.metrics = c.compute_metrics(failed, p);
        out.raw_score = c.aggregate_score(out.metrics);
        out.violation_penalty = c.compute_violations(failed, p);
        out.selection_score = out.raw_score - c.lambda_v * out.violation_penalty;
        out.accepted = false;
    end
    out.duration_s = toc(timer);
    out.candidate_id = candidate_id_from_config(candidate);
end

function id = candidate_id_from_config(candidate)
    names = {'horizon_max','Wy_base','Wy_gain','Wu','Wdu_base','Wdu_gain','hessian_reg'};
    parts = cellfun(@(n) sprintf('%s=%.17g',n,double(candidate.(n))), names, 'UniformOutput', false);
    md = java.security.MessageDigest.getInstance('SHA-256'); md.update(uint8(strjoin(parts,'|')));
    bytes = typecast(md.digest(),'uint8'); id = lower(reshape(dec2hex(bytes,2).',1,[]));
end
