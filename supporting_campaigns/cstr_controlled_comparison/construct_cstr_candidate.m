function [candidate, candidate_id] = construct_cstr_candidate(baseline, sampled, space)
% Construct one complete CSTR config from the approved seven-field space.
    arguments
        baseline (1,1) struct
        sampled (1,1) struct
        space (1,1) struct
    end
    names = {space.variables.name};
    if numel(names) ~= 7 || numel(fieldnames(sampled)) ~= 7
        error('CSTRSpace:Fields', 'Exactly seven sampled fields are required.');
    end
    candidate = baseline;
    for i = 1:numel(names)
        name = names{i}; spec = space.variables(i);
        if ~isfield(sampled, name), error('CSTRSpace:Missing', 'Missing %s.', name); end
        value = sampled.(name);
        if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
            error('CSTRSpace:Value', 'Non-finite or non-scalar value for %s.', name);
        end
        if strcmp(spec.type, 'integer')
            if value ~= round(value), error('CSTRSpace:Type', '%s must be integer.', name); end
            value = double(value);
        else
            value = double(value);
        end
        if value < spec.lo || value > spec.hi
            error('CSTRSpace:Bounds', '%s is outside its declared bounds.', name);
        end
        candidate.(name) = value;
    end
    fixed = fieldnames(space.fixed_structural_flags);
    for i = 1:numel(fixed)
        if ~isequal(candidate.(fixed{i}), space.fixed_structural_flags.(fixed{i}))
            error('CSTRSpace:Structure', 'Structural flag %s is not baseline-fixed.', fixed{i});
        end
    end
    candidate_id = config_hash(candidate, names);
end

function id = config_hash(candidate, names)
    parts = cell(1,numel(names));
    for i = 1:numel(names)
        parts{i} = sprintf('%s=%.17g', names{i}, double(candidate.(names{i})));
    end
    payload = strjoin(parts, '|');
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(uint8(payload));
    bytes = typecast(md.digest(), 'uint8');
    id = lower(reshape(dec2hex(bytes,2).',1,[]));
end
