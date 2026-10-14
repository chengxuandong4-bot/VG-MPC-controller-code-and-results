function evaluate_score_only_candidate(input_path, output_path)
% Scientific MATLAB bridge for score-only VG-MPC smoke testing.
    try
        request=jsondecode(fileread(input_path)); root=fileparts(mfilename('fullpath')); deepseek_root=fileparts(root);
        addpath(fullfile(deepseek_root,'benchmark_lightweight')); addpath(fullfile(deepseek_root,'benchmark_lightweight','src'));
        c=config_lightweight('single'); p=c.init_params(); baseline=c.single.initial_candidate(p,[]);
        if strcmp(request.mode,'baseline')
            ev=evaluate_cstr_candidate(c,p,baseline); out=pack(ev,baseline,ev.selection_score,ev.selection_score,false,'baseline');
        elseif strcmp(request.mode,'candidate')
            space=jsondecode(fileread(fullfile(root,'search_space_cstr_reduced.json')));
            [candidate,id]=construct_cstr_candidate(baseline,request.parameters,space);
            ev=evaluate_cstr_candidate(c,p,candidate); ev.candidate_id=id;
            inc=struct('candidate',baseline,'candidate_id',request.incumbent_candidate_id,'q',request.incumbent_before);
            [~,u]=update_cstr_incumbent(inc,ev); out=pack(ev,candidate,u.incumbent_q_before,u.incumbent_q_after,u.updated,u.no_update_reason);
        else
            error('ScoreOnlyBridge:Mode','Unsupported bridge mode.');
        end
        out.bridge_success=true;
    catch ME
        out=struct('bridge_success',false,'error_category','matlab_bridge_failure','candidate_id','','candidate',struct(), ...
            'execution_success',false, ...
            'metrics',[],'s',-inf,'v',100,'q',-inf,'accepted',false,'incumbent_before',NaN, ...
            'incumbent_after',NaN,'incumbent_updated',false,'simulation_duration_s',0);
    end
    fid=fopen(output_path,'w','n','UTF-8'); assert(fid>0,'Cannot write bridge output.'); cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); clear cleanup;
end

function out=pack(ev,candidate,before,after,updated,reason)
    out=struct('candidate_id',ev.candidate_id,'candidate',candidate,'execution_success',ev.execution_success, ...
        'exception_message',ev.exception_message,'metrics',ev.metrics,'s',ev.raw_score,'v',ev.violation_penalty, ...
        'q',ev.selection_score,'accepted',ev.accepted,'incumbent_before',before,'incumbent_after',after, ...
        'incumbent_updated',updated,'no_update_reason',reason,'simulation_duration_s',ev.duration_s);
end
