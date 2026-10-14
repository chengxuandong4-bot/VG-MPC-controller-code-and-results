function result = run_random_search_smoke(seed, n_eval, mode)
% NON_PUBLICATION_SMOKE_TEST: seeded Random Search plumbing validation.
    if nargin < 1, seed = 20260805; end
    if nargin < 2, n_eval = 5; end
    if nargin < 3, mode = 'run'; end
    if strcmp(mode,'sample_only')
        result = sample_candidates(seed, n_eval); return;
    end
    if strcmp(mode,'reproducibility_check')
        result = write_reproducibility_check(seed, n_eval); return;
    end
    if n_eval ~= 5, error('SmokeTest:Budget', 'Primary smoke test requires exactly five evaluations.'); end
    root = fileparts(mfilename('fullpath'));
    outdir = fullfile(root,'NON_PUBLICATION_SMOKE_TEST',sprintf('random_search_seed_%d',seed));
    if exist(outdir,'dir')
        listing = dir(outdir); listing = listing(~ismember({listing.name},{'.','..'}));
        if ~isempty(listing), error('SmokeTest:OutputExists','Refusing to overwrite non-empty output directory: %s',outdir); end
    else
        mkdir(outdir);
    end
    space = jsondecode(fileread(fullfile(root,'search_space_cstr_reduced.json')));
    deepseek_root = fileparts(root); addpath(fullfile(deepseek_root,'unified_loop'));
    addpath(fullfile(deepseek_root,'benchmark_lightweight')); addpath(fullfile(deepseek_root,'benchmark_lightweight','src'));
    c = config_lightweight('single'); p = c.init_params(); baseline = c.single.initial_candidate(p,[]);
    baseline_eval = evaluate_cstr_candidate(c,p,baseline);
    incumbent = struct('candidate',baseline,'candidate_id',baseline_eval.candidate_id,'q',baseline_eval.selection_score);
    rng(seed,'twister'); samples = sample_candidates(seed,n_eval);
    jsonl = fopen(fullfile(outdir,'evaluations.jsonl'),'w','n','UTF-8'); cleanup = onCleanup(@() fclose(jsonl));
    write_json(fullfile(outdir,'baseline.json'),struct('schema_version','NON_PUBLICATION_SMOKE_TEST_v1','configuration',baseline,'evaluation',baseline_eval));
    records = cell(1,n_eval); accepted_count=0; failed_count=0; update_count=0; timer=tic;
    rng(seed,'twister');
    for k=1:n_eval
        [candidate,id] = construct_cstr_candidate(baseline,samples(k),space); %#ok<ASGLU>
        ev = evaluate_cstr_candidate(c,p,candidate); ev.candidate_id=id;
        [inc,ur] = update_cstr_incumbent(incumbent,ev); incumbent=inc;
        rec = struct('schema_version','NON_PUBLICATION_SMOKE_TEST_v1','run_id',sprintf('random_search_seed_%d',seed), ...
            'method','random_search','smoke_test',true,'seed',seed,'evaluation_index',k,'candidate_id',id, ...
            'sampled_parameters',samples(k),'candidate',candidate,'evaluation',ev,'incumbent_update',ur);
        fprintf(jsonl,'%s\n',jsonencode(rec)); records{k}=rec;
        accepted_count=accepted_count+double(ev.accepted); failed_count=failed_count+double(~ev.execution_success); update_count=update_count+double(ur.updated);
    end
    summary=struct('schema_version','NON_PUBLICATION_SMOKE_TEST_v1','run_id',sprintf('random_search_seed_%d',seed), ...
        'method','random_search','smoke_test',true,'seed',seed,'requested_evaluations',n_eval,'completed_evaluations',n_eval, ...
        'accepted_count',accepted_count,'failed_count',failed_count,'incumbent_update_count',update_count, ...
        'baseline_score',baseline_eval.selection_score,'baseline_accepted',baseline_eval.accepted,'final_incumbent_score',incumbent.q, ...
        'best_candidate_id',incumbent.candidate_id,'total_duration_s',toc(timer));
    write_json(fullfile(outdir,'summary.json'),summary); result=struct('outdir',outdir,'summary',summary,'records',{records});
end

function result=write_reproducibility_check(seed,n)
    root=fileparts(mfilename('fullpath')); outdir=fullfile(root,'NON_PUBLICATION_SMOKE_TEST',sprintf('random_search_seed_%d',seed));
    if ~exist(outdir,'dir'), error('SmokeTest:MissingOutput','Primary smoke-test output does not exist.'); end
    space=jsondecode(fileread(fullfile(root,'search_space_cstr_reduced.json'))); deepseek_root=fileparts(root);
    addpath(fullfile(deepseek_root,'benchmark_lightweight')); addpath(fullfile(deepseek_root,'benchmark_lightweight','src'));
    c=config_lightweight('single'); p=c.init_params(); baseline=c.single.initial_candidate(p,[]);
    a=sample_candidates(seed,n); b=sample_candidates(seed,n); hashes_a=cell(1,n); hashes_b=cell(1,n);
    for k=1:n, [~,hashes_a{k}]=construct_cstr_candidate(baseline,a(k),space); [~,hashes_b{k}]=construct_cstr_candidate(baseline,b(k),space); end
    matches=isequal(a,b)&&isequal(hashes_a,hashes_b);
    result=struct('schema_version','NON_PUBLICATION_SMOKE_TEST_v1','seed',seed,'count',n, ...
        'simulations_rerun',false,'ordered_hashes_first',{hashes_a},'ordered_hashes_second',{hashes_b},'exact_match',matches);
    write_json(fullfile(outdir,'reproducibility_hashes.json'),result);
end

function samples=sample_candidates(seed,n)
    root=fileparts(mfilename('fullpath')); space=jsondecode(fileread(fullfile(root,'search_space_cstr_reduced.json'))); rng(seed,'twister'); samples=repmat(struct(),1,n);
    for k=1:n
        for i=1:numel(space.variables)
            s=space.variables(i); if strcmp(s.type,'integer'), v=randi([s.lo s.hi]); elseif strcmp(s.sampling,'log_uniform'), v=exp(log(s.lo)+rand()*(log(s.hi)-log(s.lo))); else, v=s.lo+rand()*(s.hi-s.lo); end; samples(k).(s.name)=v;
        end
    end
end

function write_json(path,data)
    fid=fopen(path,'w','n','UTF-8'); assert(fid>0,'Cannot write %s',path); c=onCleanup(@()fclose(fid)); fprintf(fid,'%s',jsonencode(data,'PrettyPrint',true)); clear c;
end
