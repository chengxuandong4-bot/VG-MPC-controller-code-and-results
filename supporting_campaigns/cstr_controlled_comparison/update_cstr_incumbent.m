function [incumbent, rec] = update_cstr_incumbent(incumbent, evaluation)
% Apply the common hard-gate and strict-improvement incumbent rule.
    before = incumbent.q;
    updated = evaluation.execution_success && evaluation.accepted && evaluation.selection_score > before;
    if updated
        incumbent.candidate = evaluation.candidate;
        incumbent.q = evaluation.selection_score;
        incumbent.candidate_id = evaluation.candidate_id;
        reason = '';
    elseif ~evaluation.execution_success
        reason = 'execution_failure';
    elseif ~evaluation.accepted
        reason = 'rejected';
    else
        reason = 'not_higher_than_incumbent';
    end
    rec = struct('incumbent_q_before',before,'candidate_q',evaluation.selection_score, ...
        'accepted',evaluation.accepted,'updated',updated,'incumbent_q_after',incumbent.q, ...
        'no_update_reason',reason);
end
