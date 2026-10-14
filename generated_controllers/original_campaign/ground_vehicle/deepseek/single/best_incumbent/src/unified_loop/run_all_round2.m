function run_all_round2()
% RUN_ALL_ROUND2  Round 2: First-round generators + Second-round outer loop.
%  Each benchmark calls its first-round candidate generation logic,
%  wrapped in the second-round verification-guided adversarial loop.
%  T = 200 for single, T = 100x2 for dual.

    root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(root)));

    fprintf('=== Round 2: 200-generation campaign ===\n');
    fprintf('  Using first-round candidate generators + second-round loop\n');
    t_start = tic;

    runs = {
        'lightweight',   'single', 'dual_obj', 'dual_strategy';
        'fault_tolerant', 'single', 'dual_obj', 'dual_strategy';
        'adaptive',      'single', 'dual_obj', 'dual_strategy';
    };

    for r = 1:size(runs,1)
        bm = runs{r,1};
        for m = 1:size(runs,2)-1
            mode = runs{r, m+1};
            cfg = get_config(bm, mode);
            if isempty(cfg), continue; end

            fprintf('\n### %s :: %s ###\n', bm, mode);
            try
                run_unified_outer_loop(cfg);
                fprintf('### %s :: %s : OK ###\n', bm, mode);
            catch ME
                fprintf('### %s :: %s : FAILED - %s ###\n', bm, mode, ME.message);
            end
        end
    end

    fprintf('\n=== Done: %.1f min ===\n', toc(t_start)/60);
end


function cfg = get_config(bm, mode)
    switch bm
        case 'lightweight',   cfg = config_lightweight(mode);
        case 'fault_tolerant', cfg = config_fault_tolerant(mode);
        case 'adaptive',      cfg = config_adaptive(mode);
        otherwise, cfg = [];
    end
    if ~isempty(cfg)
        cfg.budget_T = 200;
        if ~strcmp(mode, 'single'), cfg.budget_T = 100; end
        cfg.early_stop_N = 200;  % Run to full budget
    end
end
