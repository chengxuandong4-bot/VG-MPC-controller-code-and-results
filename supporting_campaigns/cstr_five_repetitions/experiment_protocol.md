# CSTR–DeepSeek Repeated Experiment Protocol

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-03
- Verification Status: UNVERIFIED
- Version Label: cstr_deepseek_repeat_protocol_v1

## Experiment Overview

- **Title**: Repeated CSTR controller-generation experiment with DeepSeek-V4-Flash
- **Objective**: Assess run-to-run stability of the legacy CSTR single-generator, dual-objective, and warm-start dual-strategy workflows.
- **Type**: simulation
- **Replications**: 5 completed blocks; the June 2026 runs are historical and are not included in the new aggregate statistics.

## Frozen Factors

- Benchmark: CSTR lightweight QR-AMPC.
- MATLAB random state: `rng(42, 'twister')` at the start of every mode run.
- Simulation: 0.5 s sample time, 60 s horizon, original reference and uncertainty profile.
- LLM prompt construction: legacy June 2026 `llm_generate.py` implementation saved in the original generation snapshot.
- Model request: `deepseek-v4-flash`.
- API transport: DeepSeek official Anthropic-compatible Messages API.
- Thinking mode: disabled.
- Temperature: 0.3.
- Maximum output tokens: 512.
- Response format: JSON object.
- Single budget: 200 candidates.
- Dual-objective budget: 100 A/B pairs = 200 candidates.
- Dual-strategy budget: 100 A/B pairs = 200 candidates.
- No early stopping.
- MATLAB: R2026a Update 3 on the local Mac installation. R2025a was requested initially but could not pass preflight because its MathWorks Service returned error 5001.

## Replication Block

For each `rep_01` through `rep_05`:

1. Run single-generator from the nominal baseline.
2. Run dual-objective from the nominal baseline.
3. Run dual-strategy from the best accepted configuration produced by the corresponding dual-objective run.

The third step preserves the legacy warm-start/continuation behavior. It is not an equal-information comparison with the first two modes and must be described as such in the manuscript.

## Failure Rules

- An API, response-parsing, or schema failure uses the legacy fallback candidate and still consumes one candidate evaluation.
- A rejected or simulation-failed candidate still consumes one candidate evaluation.
- No failed candidate call is replaced.
- A crashed or timed-out MATLAB mode run is retained as `aborted` and is not silently resumed or replaced.
- A full rerun after a crash requires cause review and explicit approval.
- Mode-run hard timeout: 90 minutes.

## Primary Outputs

- Run-specific baseline selection score.
- Best accepted selection score.
- Baseline-relative gain.
- Best-generation index.
- Acceptance rate.
- API success and fallback rates.
- Dual-strategy incremental improvement over its dual-objective seed.

## Analysis Plan

- Report all five run-level observations.
- Report mean, standard deviation, median, minimum, and maximum.
- Report paired differences within replication blocks.
- Do not use population-level significance claims with five blocks.
- Use the run nearest the median as the main representative response plot.
- A best observed run may be shown only when explicitly labeled as such.

## Data Integrity

- Original June 2026 results are read-only inputs.
- New results use separate `rep_XX/mode` directories.
- API credentials are never copied into result artifacts.
- Prompt text, model-request metadata, raw response excerpt, parsed configuration, metrics, acceptance state, and source status are retained for every candidate.
