function save_gen_snapshot(results_dir, gen, candidate, metrics, q_score, s_score, ...
    v_penalty, accepted, updated, role, baseline_metrics, candidate_code)
% SAVE_GEN_SNAPSHOT  Save generation snapshot with full first-round compatibility.
%  Saves 7 files per generation: 3 JSON + OVERVIEW.md + README.md + run_benchmark.m + src/

    if nargin < 11, role = ''; end
    if nargin < 12, baseline_metrics = []; end
    if nargin < 13, candidate_code = ''; end

    if isempty(role), tag = sprintf('gen_%03d', gen);
    else, tag = sprintf('gen_%03d_%s', gen, role); end

    sd = fullfile(results_dir, 'generation_snapshots', tag);
    if ~exist(sd, 'dir'), mkdir(sd); end

    cname = '';
    if isstruct(candidate) && isfield(candidate, 'name'), cname = candidate.name;
    elseif ischar(candidate), cname = candidate; end

    % 1. config_snapshot.json
    try
        cfg = candidate;
        if isstruct(cfg) && isfield(cfg, 'config'), cfg = candidate.config; end
        write_json_utf8(fullfile(sd, 'config_snapshot.json'), cfg);
    catch
    end

    % 2. metrics_snapshot.json
    try
        m = struct();
        if isstruct(metrics)
            m = metrics;
        else
            m.m_vec = metrics;
        end
        m.q_score = q_score; m.s_score = s_score; m.v_penalty = v_penalty;
        m.accepted = accepted; m.updated = updated;
        write_json_utf8(fullfile(sd, 'metrics_snapshot.json'), m);
    catch
    end

    % 3. meta_snapshot.json
    try
        meta = struct();
        meta.generation = gen; meta.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        meta.name = cname; meta.q_score = q_score;
        meta.accepted = accepted; meta.updated = updated;
        if ~isempty(role), meta.source = role; end
        if isstruct(candidate)
            trace_fields = {'llm_source','llm_provider','llm_transport', ...
                'llm_requested_model', ...
                'llm_response_model','llm_request_id','llm_http_status', ...
                'llm_prompt_tokens','llm_completion_tokens','llm_total_tokens', ...
                'llm_retry_count','llm_parse_retry_count', ...
                'llm_api_calls','llm_codex_exec_calls','llm_codex_cli_version'};
            for ti = 1:numel(trace_fields)
                tf = trace_fields{ti};
                if isfield(candidate, tf)
                    meta.(tf) = candidate.(tf);
                end
            end
        end
        write_json_utf8(fullfile(sd, 'meta_snapshot.json'), meta);
    catch
    end

    % 4. OVERVIEW.md — one-line summary
    try
        acc_str = '✓ accepted'; if ~accepted, acc_str = '✗ rejected'; end
        up_str = ''; if updated, up_str = ' ★ INCUMBENT'; end
        overview = sprintf('gen %03d — %s — q=%.4f %s%s\n', gen, cname, q_score, acc_str, up_str);
        fid = fopen(fullfile(sd, 'OVERVIEW.md'), 'w', 'n', 'UTF-8');
        if fid > 0, fprintf(fid, '%s', overview); fclose(fid); end
    catch
    end

    % 5. README.md — detailed description
    try
        write_readme(sd, gen, cname, q_score, s_score, v_penalty, accepted, updated, ...
            role, metrics, baseline_metrics);
    catch
    end

    % 6. run_benchmark.m copy — reproducible
    try
        src_script = fullfile(fileparts(mfilename('fullpath')), '..', 'run_benchmark.m');
        if exist(src_script, 'file')
            copyfile(src_script, fullfile(sd, 'run_benchmark.m'));
        end
    catch
    end

    % 7. src/ copy — benchmark source
    try
        bm_src = fullfile(fileparts(mfilename('fullpath')), '..');
        src_dst = fullfile(sd, 'src');
        if ~exist(src_dst, 'dir'), mkdir(src_dst); end
        % Copy unified_loop files
        ul = fullfile(bm_src, 'unified_loop');
        if exist(ul, 'dir')
            copyfile(ul, fullfile(src_dst, 'unified_loop'));
        end
        % Copy benchmark-specific config
        config_files = dir(fullfile(bm_src, 'benchmark_*', 'config_*.m'));
        for i = 1:numel(config_files)
            pf = fullfile(config_files(i).folder, config_files(i).name);
            copyfile(pf, src_dst);
        end
    catch
    end
end


function write_readme(sd, gen, cname, q_score, s_score, v_penalty, accepted, updated, ...
    role, metrics, baseline_metrics)
    fid = fopen(fullfile(sd, 'README.md'), 'w', 'n', 'UTF-8');
    if fid < 1, return; end

    fprintf(fid, '# Generation %03d\n\n', gen);
    fprintf(fid, '- **Name**: %s\n', cname);
    fprintf(fid, '- **Role**: %s\n', role);
    fprintf(fid, '- **q_score**: %.6f\n', q_score);
    fprintf(fid, '- **s_score**: %.6f\n', s_score);
    fprintf(fid, '- **v_penalty**: %.2f\n', v_penalty);
    fprintf(fid, '- **Accepted**: %d\n', accepted);
    fprintf(fid, '- **Updated Incumbent**: %d\n', updated);
    fprintf(fid, '- **Timestamp**: %s\n', datestr(now));

    if ~isempty(metrics)
        fprintf(fid, '\n## Metrics\n\n');
        if isstruct(metrics)
            fn = fieldnames(metrics);
            for i = 1:min(numel(fn), 20)
                val = metrics.(fn{i});
                if isnumeric(val) && isscalar(val)
                    fprintf(fid, '- **%s**: %.6f\n', fn{i}, val);
                end
            end
        elseif isnumeric(metrics)
            for i = 1:numel(metrics)
                fprintf(fid, '- **M%d**: %.6f\n', i, metrics(i));
            end
        end
    end

    if ~isempty(baseline_metrics)
        fprintf(fid, '\n## Baseline Comparison\n\n');
        if isstruct(metrics) && isstruct(baseline_metrics)
            fn = fieldnames(metrics);
            for i = 1:min(numel(fn), 10)
                fn_i = fn{i};
                if isfield(baseline_metrics, fn_i)
                    b = baseline_metrics.(fn_i);
                    m = metrics.(fn_i);
                    if isnumeric(b) && isnumeric(m) && b ~= 0
                        chg = (b - m) / abs(b) * 100;
                        fprintf(fid, '- **%s**: baseline=%.4f → current=%.4f (%+.1f%%)\n', fn_i, b, m, chg);
                    end
                end
            end
        end
    end
    fclose(fid);
end


function write_json_utf8(path, data)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    if fid < 1, return; end
    try
        fprintf(fid, '%s', jsonencode(data));
    catch
        fprintf(fid, '{}\n');
    end
    fclose(fid);
end
