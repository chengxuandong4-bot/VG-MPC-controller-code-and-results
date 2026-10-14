function preflight_cstr_repeat()
% Offline preflight. This function performs no LLM API request.

    protocol_dir = fileparts(mfilename('fullpath'));
    repeat_root = fileparts(protocol_dir);
    deepseek_root = fileparts(repeat_root);

    addpath(fullfile(deepseek_root, 'benchmark_lightweight'));
    addpath(fullfile(deepseek_root, 'benchmark_lightweight', 'src'));

    rng(42, 'twister');
    c = config_lightweight('single');
    p = c.init_params();
    baseline_result = c.run_baseline(p);
    baseline_metrics = c.compute_metrics(baseline_result, p);
    baseline_q = c.aggregate_score(baseline_metrics);

    old_cfg_path = fullfile(deepseek_root, 'LLM', 'cstr_single_llm_v2', ...
        'generation_snapshots', 'gen_196_S', 'config_snapshot.json');
    assert(exist(old_cfg_path, 'file') == 2, 'Missing old reference candidate');
    old_cfg = jsondecode(fileread(old_cfg_path));
    old_result = c.simulate_closed_loop(p, old_cfg);
    old_metrics = c.compute_metrics(old_result, p);
    old_q = c.aggregate_score(old_metrics) - ...
        c.lambda_v * c.compute_violations(old_result, p);
    old_accepted = c.acceptance_gate(old_metrics, old_result, p);

    report = struct();
    report.timestamp = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    report.matlab_version = version;
    report.baseline_q = baseline_q;
    report.reference_candidate_q = old_q;
    report.reference_candidate_accepted = old_accepted;
    report.status = 'passed';

    out_path = fullfile(repeat_root, 'preflight_report.json');
    fid = fopen(out_path, 'w', 'n', 'UTF-8');
    assert(fid > 0, 'Cannot write preflight report');
    fprintf(fid, '%s', jsonencode(report, 'PrettyPrint', true));
    fclose(fid);

    fprintf('PREFLIGHT PASSED baseline_q=%.6f reference_q=%.6f accepted=%d\n', ...
        baseline_q, old_q, old_accepted);
end
