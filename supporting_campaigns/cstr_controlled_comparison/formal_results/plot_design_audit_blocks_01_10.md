# Plot-design audit: CSTR Formal Experiment Blocks 1-10

This is a read-only derived audit. It does not rerun experiments, alter frozen outputs, change the statistical protocol, or add inferential tests.

=== A. DATA SCHEMA ===

Frozen file: `FORMAL_EXPERIMENT/formal_analysis_blocks_01_10.json`.

- Top level: `root["analysis"]`, `root["analysis_source_hash"]`, `root["validated_blocks"]`.
- Block audit data: `root["validated_blocks"]["audits"][i]`; keys are `audit_contract_version`, `block_directory`, `block_id`, `decisions`, `endpoints`, `passed`, `protocol_fingerprint`.
- Compact block-method endpoint data: `root["validated_blocks"]["endpoint_table"][i][method]`.
- Method values are exactly `random_search`, `tpe`, `score_only_vgmpc`, and `full_feedback_vgmpc`.
- Formal improvement fields: `...endpoint_table[i][method]["absolute_improvement"]`, `relative_improvement`, and `normalized_best_so_far_auc`.
- Frozen score fields: `baseline_selection_score_nonruntime` and `final_incumbent_selection_score_nonruntime` inside each endpoint.
- Per-block summary is saved twice: `validated_blocks.endpoint_table[i]` and `validated_blocks.audits[i].endpoints`.
- Statistical results are saved under `root["analysis"]["friedman"]`, `posthoc`, and `method_descriptives`.
- Raw 3,200 evaluations are NOT embedded in the frozen JSON. Their source is each included audit's `block_directory/evaluations.jsonl`.
- Raw evaluation paths: `row["block_id"]`, `block_seed`, `method`, `global_slot_index`, `method_slot_index`, `round_index`, `order_cycle`, `order_position`.
- Raw scores/objectives: `row["raw_score_nonruntime"]`, `selection_score_nonruntime`, `optimizer_objective`, `normalized_optimizer_score`, and incumbent score fields ending in `_before`/`_after`.
- Candidate score is `row["selection_score_nonruntime"]`; common baseline is `common_baseline.json["selection_score_nonruntime"]`.
- Success/failure: `row["execution_success"]`, `result_failed`, `simulation_failure`, `proposal_valid`, and `proposal_failure_category`; selected blocks also audit `abort_report_absent`.
- Runtime fields: `row["metrics"]["avg_controller_time_ms"]` (frozen runtime metric), plus engineering durations `proposal_duration`, `api_duration`, `simulation_duration`, and `total_slot_duration`.
- Cache hits intentionally have `avg_controller_time_ms = null`, `runtime_remeasured = false`, and `runtime_measurement_origin = not_remeasured_cache_hit`.
- No separate timeout field exists in successful rows. A block-level timeout would be recorded in `abort_report.json`; all ten included blocks have no abort report.

=== B. FORMAL STATISTICAL DESIGN ===

- Primary endpoint: `final_absolute_nonruntime_score_improvement_after_80_slots`.
- Independent repeated unit: one completed formal block. There are 10 paired blocks.
- Within each block, the four methods share the block context/common baseline and each receives 80 method slots. The formal pair is the block-level final endpoint, not individual evaluation position.
- The 3,200 evaluations are not treated as independent replicates. Same-position rows are schedule-balanced observations, not the inferential pairing unit.
- No block was dropped and no imputation was used.
- The 40-row descriptive table below defines `median/mean/std/q1/q3/min/max_improvement` as the 80-slot incumbent best-so-far trajectory (`incumbent_selection_score_nonruntime_after - common baseline`). These are descriptive and are not the Friedman inputs.
- `std_improvement` is population SD over the complete 80-slot trajectory.

=== C. FRIEDMAN RESULT ===

- Input unit: 10 block-level final absolute improvements per method; matrix shape 10 x 4.
- Grouping variable: method. Blocking/paired variable: block_id. Rank sums in protocol order `('random_search', 'tpe', 'score_only_vgmpc', 'full_feedback_vgmpc')`: [31.0, 39.0, 10.0, 20.0].
- Statistic = 28.919999999999987; df = 3; p = 2.327806591795379e-06; alpha = 0.05; rejected = True.
- Calculation: custom Python implementation with within-block average ranks, tie correction, and chi-square(3) asymptotic survival function. It does not call SciPy/R/MATLAB.
- Code path: `formal_analysis_contract.py::friedman_test` (around line 262), called by `run_frozen_analysis` (around line 338).

=== D. SIX PAIRWISE COMPARISONS ===

Differences are Method A minus Method B. The formal JSON does not serialize a test statistic; the audit reports the implied minimum signed-rank sum W from the same 10 values without changing the test result.

| Method A | Method B | Test | n | Paired | Statistic | Raw p | Holm p | Median diff | Effect size | A>B/Tie/A<B |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Search | TPE | Exact signed-rank permutation | 10 | yes | not serialized; implied W=1 | 0.00390625 | 0.01171875 | -0.160601454 | -0.963636364 | 1/0/9 |
| Random Search | Score-only VG-MPC | Exact signed-rank permutation | 10 | yes | not serialized; implied W=0 | 0.001953125 | 0.01171875 | 0.0189364305 | 1 | 10/0/0 |
| Random Search | Full-feedback VG-MPC | Exact signed-rank permutation | 10 | yes | not serialized; implied W=0 | 0.001953125 | 0.01171875 | 0.0152577918 | 1 | 10/0/0 |
| TPE | Score-only VG-MPC | Exact signed-rank permutation | 10 | yes | not serialized; implied W=0 | 0.001953125 | 0.01171875 | 0.173880423 | 1 | 10/0/0 |
| TPE | Full-feedback VG-MPC | Exact signed-rank permutation | 10 | yes | not serialized; implied W=0 | 0.001953125 | 0.01171875 | 0.172232357 | 1 | 10/0/0 |
| Score-only VG-MPC | Full-feedback VG-MPC | Exact signed-rank permutation | 10 | yes | not serialized; implied W=0 | 0.001953125 | 0.01171875 | -0.00338653133 | -1 | 0/0/10 |

Implementation details:

- `formal_analysis_contract.py::wilcoxon_pratt` ranks absolute paired block differences. With zero differences absent and no absolute-difference ties, it enumerates all `2^10` sign assignments and returns a two-sided exact p-value.
- Effect size is `(positive rank sum - negative rank sum) / nonzero rank sum`, already serialized in the frozen JSON.
- Five raw p-values equal `1/512 = 0.001953125`; Random Search vs TPE equals `1/256 = 0.00390625` because implied W is 1 rather than 0.
- Holm is standard step-down monotone adjustment in `holm_adjust`. The smallest raw p multiplied by 6 is 0.01171875; the cumulative-max monotonicity rule propagates that value through all later ordered hypotheses, so all six adjusted p-values are exactly 0.01171875. They are not all equal because all raw p-values are equal.

=== E. 10x4 BLOCK-METHOD SUMMARY ===

Complete 40-row table (runtime is measured controller runtime only; null cache-hit runtimes are excluded):

| Block | Seed | Method | n | Success | Rate | Final endpoint | Median traj | Mean traj | SD traj | Q1 | Q3 | Min | Max | Runtime n | Median ms | Mean ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20261001 | TPE | 80 | 80 | 1 | 0.189559863 | 0.0277951089 | 0.0780385256 | 0.0760901351 | 0.0249298151 | 0.180450844 | 0 | 0.189559863 | 80 | 9.26066441 | 10.2467841 |
| 1 | 20261001 | Random Search | 80 | 80 | 1 | 0.0105470729 | 0.0105470729 | 0.0104860965 | 0.000196912803 | 0.0105470729 | 0.0105470729 | 0.00985020015 | 0.0105470729 | 80 | 10.2872141 | 11.2988226 |
| 1 | 20261001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.00319847442 | 0.00168113193 | 0.00228806892 | 0.000743342977 | 0.00168113193 | 0.00319847442 | 0.00168113193 | 0.00319847442 | 8 | 13.4984132 | 13.7256042 |
| 1 | 20261001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 10.3973182 | 10.4584269 |
| 2 | 20262001 | TPE | 80 | 80 | 1 | 0.198319059 | 0.196464635 | 0.11811596 | 0.093099159 | 0.00628204629 | 0.196464635 | 5.80560118e-05 | 0.198319059 | 80 | 8.49810226 | 8.89853095 |
| 2 | 20262001 | Random Search | 80 | 80 | 1 | 0.008954948 | 0.008954948 | 0.0056964151 | 0.00406730704 | 0.000566578355 | 0.008954948 | 0 | 0.008954948 | 80 | 10.5846033 | 11.2886885 |
| 2 | 20262001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.00357458824 | 0.00357458824 | 0.00323904972 | 0.000829953321 | 0.0035098559 | 0.00357458824 | 0 | 0.00357458824 | 10 | 13.8127234 | 14.0866639 |
| 2 | 20262001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 9.84077743 | 10.0579081 |
| 3 | 20263001 | TPE | 80 | 80 | 1 | 0.204101774 | 0.159869775 | 0.102563016 | 0.0946436239 | 0.00159033294 | 0.204101774 | 0 | 0.204101774 | 80 | 8.44088819 | 8.90768672 |
| 3 | 20263001 | Random Search | 80 | 80 | 1 | 0.0191226209 | 0.0191226209 | 0.0105517089 | 0.00878817856 | 0.0014706264 | 0.0191226209 | 0.0014706264 | 0.0191226209 | 80 | 10.6417785 | 11.5972332 |
| 3 | 20263001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.00552449305 | 0.00340319563 | 0.00357710003 | 0.0011843135 | 0.00321648576 | 0.00340319563 | 0 | 0.00552449305 | 24 | 14.2569951 | 13.8336931 |
| 3 | 20263001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 10.2771781 | 11.1279615 |
| 4 | 20264001 | TPE | 80 | 80 | 1 | 0.160012043 | 0.130467111 | 0.110966613 | 0.0624098869 | 0.100007458 | 0.160012043 | 1.77793888e-05 | 0.160012043 | 80 | 9.58196649 | 10.1287691 |
| 4 | 20264001 | Random Search | 80 | 80 | 1 | 0.00844702511 | 0.00844702511 | 0.00501664362 | 0.0037267561 | 0.00112842504 | 0.00844702511 | 0 | 0.00844702511 | 80 | 11.6045545 | 12.2544897 |
| 4 | 20264001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.000943855816 | 0.000943855816 | 0.00053416769 | 0.000443020495 | 3.90788405e-05 | 0.000943855816 | 0 | 0.000943855816 | 27 | 13.0990059 | 14.0220872 |
| 4 | 20264001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 10.8401542 | 10.8911975 |
| 5 | 20265001 | TPE | 80 | 80 | 1 | 0.037308182 | 0.0193753746 | 0.0189637448 | 0.00540310994 | 0.0186725263 | 0.0222300868 | 0 | 0.037308182 | 80 | 13.6961927 | 13.4824422 |
| 5 | 20265001 | Random Search | 80 | 80 | 1 | 0.0408382566 | 0.0408382566 | 0.0265332904 | 0.0175876751 | 0.00624718908 | 0.0408382566 | 0 | 0.0408382566 | 80 | 11.8118592 | 13.0194023 |
| 5 | 20265001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.0196497146 | 0.0196497146 | 0.0108529685 | 0.0097251872 | 0.000114662415 | 0.0196497146 | 0 | 0.0196497146 | 10 | 15.7149056 | 15.5194435 |
| 5 | 20265001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 10.8272073 | 11.0804313 |
| 6 | 20266001 | TPE | 80 | 80 | 1 | 0.147106638 | 0.0101644321 | 0.0586936804 | 0.0686361039 | 0.00112589378 | 0.147106638 | 0 | 0.147106638 | 80 | 10.0926222 | 10.8838239 |
| 6 | 20266001 | Random Search | 80 | 80 | 1 | 0.038453146 | 0.038453146 | 0.0357086071 | 0.00964141479 | 0.038453146 | 0.038453146 | 0 | 0.038453146 | 80 | 11.6556835 | 12.5918084 |
| 6 | 20266001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.00574937453 | 0.00554852109 | 0.00435592226 | 0.0019234271 | 0.00307051038 | 0.00554852109 | 0 | 0.00574937453 | 33 | 12.150159 | 12.292844 |
| 6 | 20266001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 11.3190819 | 11.2913655 |
| 7 | 20267001 | TPE | 80 | 80 | 1 | 0.172651695 | 0.170379056 | 0.146223926 | 0.0580549726 | 0.164804442 | 0.170379056 | 2.848699e-05 | 0.172651695 | 80 | 9.40457569 | 10.3074643 |
| 7 | 20267001 | Random Search | 80 | 80 | 1 | 0.00408358626 | 0.00408358626 | 0.00370543204 | 0.00113503692 | 0.00408358626 | 0.00408358626 | 0 | 0.00408358626 | 80 | 11.8944378 | 12.5611852 |
| 7 | 20267001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.000326616728 | 0.000326616728 | 0.00021992072 | 0.000146437568 | 2.99436834e-05 | 0.000326616728 | 0 | 0.000326616728 | 20 | 12.9451786 | 12.8899849 |
| 7 | 20267001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 10.8779858 | 11.2361213 |
| 8 | 20268001 | TPE | 80 | 80 | 1 | 0.175109151 | 0.145837726 | 0.110043362 | 0.0701709524 | 0.0033263243 | 0.150078919 | 0 | 0.175109151 | 80 | 9.20493785 | 9.67523003 |
| 8 | 20268001 | Random Search | 80 | 80 | 1 | 0.0345315898 | 0.0339726057 | 0.0291104386 | 0.0120671833 | 0.0339726057 | 0.0345315898 | 0 | 0.0345315898 | 80 | 11.6557017 | 11.870636 |
| 8 | 20268001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.002969515 | 0.002969515 | 0.00269347387 | 0.000748798609 | 0.00292684482 | 0.002969515 | 0 | 0.002969515 | 19 | 18.076774 | 16.6093004 |
| 8 | 20268001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 11.2522896 | 10.9115805 |
| 9 | 20269001 | TPE | 80 | 80 | 1 | 0.171385039 | 0.153396948 | 0.137052028 | 0.0429415363 | 0.100222693 | 0.161344367 | 0 | 0.171385039 | 80 | 8.75484062 | 9.59064053 |
| 9 | 20269001 | Random Search | 80 | 80 | 1 | 0.01875024 | 0.01875024 | 0.0143898987 | 0.0071471263 | 0.0062507291 | 0.01875024 | 0 | 0.01875024 | 80 | 11.7051536 | 12.2617824 |
| 9 | 20269001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.000146017216 | 0.000130325243 | 0.000124179422 | 2.21568039e-05 | 9.27788229e-05 | 0.000146017216 | 9.27788229e-05 | 0.000146017216 | 21 | 15.5493979 | 15.6995121 |
| 9 | 20269001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 10.6871882 | 11.0057566 |
| 10 | 20270001 | TPE | 80 | 80 | 1 | 0.270671088 | 0.160537978 | 0.149913573 | 0.109431989 | 0.0205254879 | 0.270671088 | 0 | 0.270671088 | 80 | 9.08959687 | 10.2787492 |
| 10 | 20270001 | Random Search | 80 | 80 | 1 | 0.0208222122 | 0.0208222122 | 0.0163101851 | 0.00643638917 | 0.00942991313 | 0.0208222122 | 0.00103435464 | 0.0208222122 | 80 | 10.6163549 | 11.4202149 |
| 10 | 20270001 | Full-feedback VG-MPC | 80 | 80 | 1 | 0.00390475634 | 0.00321648576 | 0.00293470153 | 0.00132950658 | 0.00321648576 | 0.00390475634 | 0 | 0.00390475634 | 18 | 12.2915276 | 12.4796337 |
| 10 | 20270001 | Score-only VG-MPC | 80 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 10.4457309 | 10.7150176 |

Formal 10x4 final absolute-improvement matrix (this is the actual inferential input; each cell is one final endpoint, not a median):

| Block | TPE | Random | Full-feedback | Score-only |
|---|---|---|---|---|
| 1 | 0.189559863 | 0.0105470729 | 0.00319847442 | 0 |
| 2 | 0.198319059 | 0.008954948 | 0.00357458824 | 0 |
| 3 | 0.204101774 | 0.0191226209 | 0.00552449305 | 0 |
| 4 | 0.160012043 | 0.00844702511 | 0.000943855816 | 0 |
| 5 | 0.037308182 | 0.0408382566 | 0.0196497146 | 0 |
| 6 | 0.147106638 | 0.038453146 | 0.00574937453 | 0 |
| 7 | 0.172651695 | 0.00408358626 | 0.000326616728 | 0 |
| 8 | 0.175109151 | 0.0345315898 | 0.002969515 | 0 |
| 9 | 0.171385039 | 0.01875024 | 0.000146017216 | 0 |
| 10 | 0.270671088 | 0.0208222122 | 0.00390475634 | 0 |

Requested 10x4 median-improvement matrix (descriptive median of 80-slot incumbent trajectories):

| Block | TPE | Random | Full-feedback | Score-only |
|---|---|---|---|---|
| 1 | 0.0277951089 | 0.0105470729 | 0.00168113193 | 0 |
| 2 | 0.196464635 | 0.008954948 | 0.00357458824 | 0 |
| 3 | 0.159869775 | 0.0191226209 | 0.00340319563 | 0 |
| 4 | 0.130467111 | 0.00844702511 | 0.000943855816 | 0 |
| 5 | 0.0193753746 | 0.0408382566 | 0.0196497146 | 0 |
| 6 | 0.0101644321 | 0.038453146 | 0.00554852109 | 0 |
| 7 | 0.170379056 | 0.00408358626 | 0.000326616728 | 0 |
| 8 | 0.145837726 | 0.0339726057 | 0.002969515 | 0 |
| 9 | 0.153396948 | 0.01875024 | 0.000130325243 | 0 |
| 10 | 0.160537978 | 0.0208222122 | 0.00321648576 | 0 |

Success-rate matrix: all 40 cells are `80/80 = 1.0`. Overall, every method is `800/800 = 1.0`.

=== F. FULL-FEEDBACK VS SCORE-ONLY ===

| Block | Full-feedback | Score-only | Full - Score | Winner |
|---|---|---|---|---|
| 1 | 0.00319847442 | 0 | 0.00319847442 | Full-feedback |
| 2 | 0.00357458824 | 0 | 0.00357458824 | Full-feedback |
| 3 | 0.00552449305 | 0 | 0.00552449305 | Full-feedback |
| 4 | 0.000943855816 | 0 | 0.000943855816 | Full-feedback |
| 5 | 0.0196497146 | 0 | 0.0196497146 | Full-feedback |
| 6 | 0.00574937453 | 0 | 0.00574937453 | Full-feedback |
| 7 | 0.000326616728 | 0 | 0.000326616728 | Full-feedback |
| 8 | 0.002969515 | 0 | 0.002969515 | Full-feedback |
| 9 | 0.000146017216 | 0 | 0.000146017216 | Full-feedback |
| 10 | 0.00390475634 | 0 | 0.00390475634 | Full-feedback |

- Full-feedback better: 10/10; ties: 0/10; Score-only better: 0/10.
- Descriptive paired median difference = 0.00338653133; IQR = 0.00366928826; positive/zero/negative = 10/0/0.
- Natural pair: block-level final endpoint under the same block/common baseline and equal 80-slot budget. Method-slot positions are not separate independent pairs and method RNG/generation streams are method-specific.
- Formal stored comparison is Score-only minus Full-feedback: median difference -0.00338653133, raw p 0.001953125, Holm p 0.01171875, effect size -1.0.

=== G. TPE / RANDOM VS LLM DESCRIPTIVE GAPS ===

| A | B | Median A-B | Ratio of medians | Min gap | Max gap | A>B | Tie | A<B |
|---|---|---|---|---|---|---|---|---|
| TPE | Full-feedback VG-MPC | 0.172232357 | 51.3446963 | 0.0176584674 | 0.266766331 | 10 | 0 | 0 |
| Random Search | Full-feedback VG-MPC | 0.0152577918 | 5.59168914 | 0.00375696954 | 0.0327037714 | 10 | 0 | 0 |
| TPE | Score-only VG-MPC | 0.173880423 | NOT AVAILABLE | 0.037308182 | 0.270671088 | 10 | 0 | 0 |
| Random Search | Score-only VG-MPC | 0.0189364305 | NOT AVAILABLE | 0.00408358626 | 0.0408382566 | 10 | 0 | 0 |

Ratios against Score-only are NOT AVAILABLE because its median is zero; infinity would be meaningless.

=== H. DISTRIBUTION CHARACTERISTICS ===

| Method | n blocks | Min | Q1 | Median | Mean | Q3 | Max | SD | Moment skew | Zeros | Negative | Tukey outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TPE | 10 | 0.037308182 | 0.162855292 | 0.173880423 | 0.172622453 | 0.19612926 | 0.270671088 | 0.0583851396 | -0.883060355 | 0 | 0 | 0.037308182, 0.270671088 |
| Random Search | 10 | 0.00408358626 | 0.00935297923 | 0.0189364305 | 0.0204550698 | 0.0311042454 | 0.0408382566 | 0.0132707278 | 0.424346526 | 0 | 0 | none |
| Full-feedback VG-MPC | 10 | 0.000146017216 | 0.00145027061 | 0.00338653133 | 0.00459874059 | 0.00511955887 | 0.0196497146 | 0.00564007868 | 2.0813162 | 0 | 0 | 0.0196497146 |
| Score-only VG-MPC | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NOT AVAILABLE | 10 | 0 | none |

- Formal endpoints contain no negative values. Score-only is a 10/10 point mass at zero.
- TPE is roughly 9.18x Random Search and 51.34x Full-feedback by median. A single ordinary linear axis compresses both LLM methods near zero.
- Full-feedback is strongly right-skewed because Block 5 is much larger than its other blocks. TPE has a low Block 5 and high Block 10 Tukey outlier; the method ordering changes only in Block 5, where Random Search exceeds TPE.
- A conventional log axis is invalid for Score-only zeros and for zero-valued trajectory points. Adding an arbitrary epsilon would be post hoc and scale-dependent.
- A broken axis is not recommended because it visually exaggerates the smaller methods and conflicts with the stated non-superiority positioning.
- The appropriate raw points for primary figures are the 40 block-level endpoints. Plotting all 3,200 evaluation rows as independent points would be pseudoreplication; raw trajectories belong in a supplementary convergence plot.

=== I. FIGURE 1 RECOMMENDATION ===

- Recommended graph: two aligned panels forming one figure. Panel A: horizontal boxplot plus all 10 jittered block endpoints for all four methods on a common linear scale starting at zero. Panel B: explicit magnified linear panel for Random Search, Full-feedback, and Score-only, also starting at zero; label it as a magnified view, not a broken axis.
- Not recommended: violin plot (n=10 per method is too small and Score-only is degenerate), 3,200-point raw-evaluation boxplot, log scale, epsilon-log transform, or broken axis.
- x-axis: final absolute non-runtime score improvement after 80 slots. y-axis: method. Unit: one block endpoint. Show all 10 points, median, and IQR; no distributional smoothing.
- ECDF is a defensible secondary choice but is stepwise at n=10 and less direct for the primary comparison.

=== J. FIGURE 2 RECOMMENDATION ===

- Recommended: 10x4 block-by-method rank heatmap, cells annotated with the exact final absolute improvement. Color encodes within-block rank; annotation preserves magnitude.
- Second choice: small-multiple block endpoint dot plots with one panel per method and independently labeled linear scales. Do not connect blocks as a time series.
- Not recommended: four raw-value lines on one y-axis (compresses three methods and falsely implies temporal continuity), or rank-only lines without numeric annotations (discard effect magnitude).
- Interaction: ordinal pattern is stable in 9/10 blocks; Block 5 swaps TPE and Random Search. Magnitudes vary substantially, especially Full-feedback Block 5 and TPE Blocks 5/10.

=== K. FIGURE 3 RECOMMENDATION ===

- Recommended: a paired block-level dot/slope panel for Full-feedback versus Score-only plus an aligned difference dot plot (`Full - Score`) for the same 10 blocks.
- Pairing is valid at block level. Show all block IDs. Because every Score-only endpoint is zero, a simple two-group boxplot or raincloud hides pairing and a slope plot alone suffers complete overlap at the Score-only side; the difference panel is essential.
- Annotate the frozen pairwise result only: exact signed-rank raw p 0.001953125, Holm p 0.01171875, and median Full-minus-Score difference 0.00338653133. Do not add a new test.

=== L. SUCCESS-RATE FIGURE DECISION ===

- Overall and every block-method cell are 100% execution success (3,200/3,200 successful included evaluations). A success-rate figure has no variation and should NOT enter the main text.
- Put one sentence or a supplementary table entry instead. Note that cache hits count as successful evaluations; success rate is not proposal uniqueness or simulation count.

=== M. RUNTIME FIGURE DECISION ===

| Method | Measured n | Median ms | Q1 | Q3 | Mean ms |
|---|---|---|---|---|---|
| TPE | 800 | 9.29596042 | 8.52529818 | 11.1831855 | 10.2400121 |
| Random Search | 800 | 11.2509717 | 9.35447656 | 13.9870437 | 12.0164263 |
| Full-feedback VG-MPC | 190 | 13.1639075 | 11.8973277 | 15.4189233 | 13.9464964 |
| Score-only VG-MPC | 39 | 10.7413028 | 10.2911675 | 11.4772472 | 10.9008606 |

- Runtime is excluded from score and should remain an auxiliary engineering metric.
- Measurement availability is structurally unequal: TPE and Random have 800 measured simulations each, Full-feedback 190, Score-only 39, because duplicate cache hits are deliberately not remeasured.
- The observed controller-runtime differences are modest relative to the missingness/design asymmetry. Do not use a main-text runtime figure or combine runtime with score. Use a supplementary table with measured n, median, and IQR.
- `total_slot_duration` is available but mixes API latency, proposal time, simulation, and cache behavior; it is a different engineering quantity and should not be substituted silently.

=== N. FINAL 3-FIGURE PAPER RECOMMENDATION ===

Figure 1 - Final improvement across four methods

- Research question: RQ2. Data: 40 formal block endpoints. Aggregation: one final absolute improvement per block-method after 80 slots. Marks: horizontal box + 10 raw dots; aligned all-method and magnified non-TPE panels. Uncertainty: IQR; no model-based CI. Statistical annotation: Friedman result only.
- Supports: numerical/black-box baselines provide context and TPE is strongest in 9/10 blocks. Does not support: 40 independent samples or LLM superiority over TPE.
- Limitation: n=10 blocks and strongly separated scales.

Figure 2 - Consistency of method ordering across formal blocks

- Research question: RQ3. Data: same 40 final endpoints. Aggregation: within-block ranks, with exact endpoint annotations. Axes: block ID x method. Marks: rank heatmap. Uncertainty: none; every block is shown. Statistical annotation: none beyond optional Friedman p in caption.
- Supports: ordering is stable in 9/10 blocks and not driven solely by one block. Does not support: equal effect magnitude across blocks or a temporal trend.
- Limitation: rank colors compress magnitude; numeric annotations and Figure 1 are required.

Figure 3 - Effect of rich feedback within LLM-based optimization

- Research question: RQ1. Data: 10 paired Full-feedback and Score-only final endpoints, plus 10 paired differences. Aggregation: block. Marks: paired dots/slopes + aligned difference dots. Uncertainty: median and IQR of differences; all raw blocks shown. Statistical annotation: frozen exact signed-rank raw/Holm p and effect size.
- Supports: under this frozen protocol, Full-feedback exceeds Score-only in all 10 blocks. Does not support: LLM beats TPE, feedback alone causally explains all behavior outside this protocol, or practical superiority proportional to statistical significance.
- Limitation: Score-only is degenerate at zero and n=10 is small.

Optional supplementary figure: four method-specific convergence small multiples, x = method slot 1-80, y = incumbent absolute improvement, showing all 10 block trajectories plus their median trajectory. This is more informative than success-rate or runtime plots and must be labeled descriptive/non-independent within block.

=== O. STATISTICAL / VISUALIZATION WARNINGS ===

1. The inferential n is 10 paired blocks, not 3,200 evaluations and not 800 method rows.
2. Per-slot incumbent trajectories are autocorrelated, monotone best-so-far sequences and must not be used as independent inferential samples.
3. Score-only is exactly zero in all 10 formal endpoints; log axes and ratios with Score-only denominators are undefined.
4. Do not add epsilon before logging, delete Block 5/10 as outliers, truncate axes, or select a scale to make an LLM method appear competitive with TPE.
5. Full-feedback > Score-only is stable but small in absolute magnitude relative to TPE; statistical significance is not practical equivalence or superiority.
6. All Holm-adjusted p-values matching is a consequence of discrete exact p-values plus the standard cumulative-max Holm implementation, not evidence that all contrasts have identical effects.
7. The pairwise test statistic is not serialized in the frozen output. Any reported W must be labeled reconstructed/implied, while p-values/effect sizes should be quoted from the frozen JSON.
8. Success is 100%, but cache-hit success is not a fresh simulation and not a unique proposal.
9. Runtime missingness is protocol-driven and strongly method-dependent; runtime comparisons are descriptive only.
10. Block ID is a replicate label, not chronological time; line charts across Blocks 1-10 can imply a false trend.
11. Rank plots support consistency claims but erase magnitude; pair them with exact values or Figure 1.
12. The formal endpoint is final best-so-far improvement after equal budget. Candidate-score distributions answer a different question and should not replace it post hoc.

Derived files:

- `plot_design_audit_blocks_01_10.md`
- `plot_design_block_method_summary.csv`
- `plot_design_pairwise_summary.csv`
