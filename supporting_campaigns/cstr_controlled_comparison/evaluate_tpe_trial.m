function evaluate_tpe_trial(input_path, output_path)
% MATLAB scientific bridge for the NON_PUBLICATION_SMOKE_TEST TPE runner.
    try
        request = jsondecode(fileread(input_path));
        root = fileparts(mfilename('fullpath')); deepseek_root = fileparts(root);
        addpath(fullfile(deepseek_root,'unified_loop'));
        addpath(fullfile(deepseek_root,'benchmark_lightweight'));
        addpath(fullfile(deepseek_root,'benchmark_lightweight','src'));
        c = config_lightweight('single'); p = c.init_params();
        baseline = c.single.initial_candidate(p, []);
        if strcmp(request.mode, 'baseline')
            evaluation = evaluate_cstr_candidate(c, p, baseline);
            out = standard_record('baseline', baseline, evaluation, struct('q',evaluation.selection_score));
        elseif strcmp(request.mode, 'trial')
            space = jsondecode(fileread(fullfile(root,'search_space_cstr_reduced.json')));
            [candidate, candidate_id] = construct_cstr_candidate(baseline, request.sampled_parameters, space);
            evaluation = evaluate_cstr_candidate(c, p, candidate); evaluation.candidate_id = candidate_id;
            incumbent = struct('candidate',baseline,'candidate_id',request.incumbent_candidate_id,'q',request.incumbent_before);
            [~, update] = update_cstr_incumbent(incumbent, evaluation);
            out = standard_record('trial', candidate, evaluation, update);
        else
            error('TPEBridge:Mode','Unknown bridge mode: %s', request.mode);
        end
        out.bridge_success = true;
    catch ME
        out = struct('bridge_success',false,'bridge_exception',getReport(ME,'basic'), ...
            'execution_success',false,'metrics',[],'s',-inf,'v',100,'q',-inf,'accepted',false, ...
            'incumbent_before',NaN,'incumbent_after',NaN,'incumbent_updated',false);
    end
    fid=fopen(output_path,'w','n','UTF-8'); assert(fid>0,'Cannot write bridge output.'); cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); clear cleanup;
end

function out = standard_record(kind, candidate, ev, update)
    out = struct('bridge_success',true,'kind',kind,'candidate_id',ev.candidate_id, ...
        'candidate',candidate,'execution_success',ev.execution_success,'exception_message',ev.exception_message, ...
        'metrics',ev.metrics,'s',ev.raw_score,'v',ev.violation_penalty,'q',ev.selection_score, ...
        'accepted',ev.accepted,'duration_s',ev.duration_s);
    if strcmp(kind,'baseline')
        out.incumbent_before = ev.selection_score; out.incumbent_after = ev.selection_score; out.incumbent_updated=false; out.no_update_reason='baseline';
    else
        out.incumbent_before=update.incumbent_q_before; out.incumbent_after=update.incumbent_q_after;
        out.incumbent_updated=update.updated; out.no_update_reason=update.no_update_reason;
    end
end
