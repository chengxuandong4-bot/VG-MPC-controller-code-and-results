# TinyMPC 146 s Figure-8: 200-Iteration Offline Report

Date: 2026-08-11

## Status

The deterministic 200-candidate offline search completed successfully. The
selected candidate is generation 195. It is **not flight-approved**. All full
candidate configurations are deliberately fail-closed with `audit_only=true`
and `actual_publish_enabled=false`.

## Scope and frozen safety fields

The search changed only:

- nine diagonal state weights;
- three diagonal input weights;
- three augmented-integral state limits;
- prediction horizon, constrained to 14–20 steps.

It did not change the 20 Hz period, Figure-8 reference, acceleration limits,
position/velocity safety envelope, solver tolerance, 20 ms solve-time limit,
RC policy, ROS topics, publication boundary, or flight-handover state machine.
The iteration harness compares every non-tunable field structurally before a
candidate is evaluated.

## Historical motivation

The supplied 146-second result log contains five sessions that entered ACTIVE:
three completed, one exited on a 23.7 ms TinyMPC solve, and one exited near the
end on the Y-position-error safety boundary. The objective therefore emphasizes
Y robustness and bounded numerical workload in addition to tracking error.

## Search result

Exactly 200 candidates were evaluated, plus generation 0 as the baseline.
There were 29 incumbent updates.

| Metric | Baseline | Generation 195 | Change |
|---|---:|---:|---:|
| Robust aggregate loss (lower is better) | 0.11369438 | 0.07971531 | -29.89% |
| Horizon | 20 | 14 | -30.00% |
| Workload proxy (horizon × mean iterations) | 536.26 | 126.92 | -76.34% |

Training-scenario loss reductions were 37.81% nominal, 28.98% under positive
disturbance, 30.63% under negative disturbance, 29.48% for weak Y authority,
and 26.35% for delayed underactuation. All five scenarios completed the full
146-second simulation and passed the conservative safety-margin gates.

## Paper-compatible legacy `q_score`

The old paper score has now been adapted to the 9-state model without changing
its eight weights, normalization constants, reciprocal aggregation, violation
coefficient, or higher-is-better direction:

\[
s=\frac{1}{1+\sum_j w_j M_j/n_j},\qquad q=s-0.5v.
\]

Position metrics and solve time map directly. Because the 9-state outer loop
does not contain attitude or body-rate states, achieved acceleration is lifted
through the same small-angle hover relationship used by the paper's 12-state
model (`roll=-a_y/g`, `pitch=a_x/g`, `yaw=0`). Equivalent thrust and body
moments provide the control-energy proxy. The three integral metrics are scaled
to 10-second equivalents before applying the paper's unchanged normalizers,
because the new trajectory lasts 146 seconds while the paper benchmark lasted
10 seconds.

| Legacy paper-style metric (higher is better) | Baseline | Generation 195 | Change |
|---|---:|---:|---:|
| Primary nominal `q_score` | 0.991670 | 0.993035 | +0.1376% |
| Supplemental five-scenario mean `q_score` | 0.984611 | 0.985614 | +0.1019% |
| Supplemental worst-scenario `q_score` | 0.981611 | 0.981849 | +0.0243% |

The primary paper-style score is the nominal single-scenario value; robust mean
and worst-case values are supplemental and are not folded into it. Both configs
pass all five hard gates. The adapted score is near its upper bound because the
legacy drone normalizers are loose relative to this low-error outer-loop task.
Consequently, the nominal normalized cost decreases by about 16.50%, while the
reciprocal `q_score` increases by only 0.1376%.

This adapter establishes a consistent baseline-relative `q_score` inside the
new 9-state benchmark. Its absolute value must not be compared directly with
the paper's legacy 12-state score table: the plant, reference, control interface,
sample period, and test duration remain different. Claiming direct absolute
cross-benchmark comparability would be methodologically incorrect.

## Selected controller parameters

| Parameter | Baseline | Generation 195 |
|---|---:|---:|
| Q position X | 15.0000 | 33.5126 |
| Q position Y | 15.0000 | 45.0000 |
| Q position Z | 25.0000 | 26.7980 |
| Q velocity X | 4.0000 | 2.1995 |
| Q velocity Y | 4.0000 | 2.1294 |
| Q velocity Z | 6.0000 | 4.4441 |
| Q integral X | 0.5000 | 0.8119 |
| Q integral Y | 8.0000 | 11.0533 |
| Q integral Z | 12.0000 | 23.8292 |
| R acceleration X | 2.0000 | 2.1550 |
| R acceleration Y | 2.0000 | 2.4029 |
| R acceleration Z | 3.0000 | 2.2509 |
| Integral limit X (m·s) | 0.1000 | 0.08520 |
| Integral limit Y (m·s) | 0.1500 | 0.10660 |
| Integral limit Z (m·s) | 0.1200 | 0.08089 |

## Independent validation

Twenty-four deterministic holdout scenarios were generated with unseen
combinations of plant gain, actuator lag, 0–2 control-step delay, drag, constant
acceleration bias, initial position error, and initial velocity.

- baseline gates: 24/24;
- candidate gates: 24/24;
- candidate improved cases: 24/24;
- holdout mean loss reduction: 27.89%.

In separate 10,000-solve augmented-state numerical benchmarks, both controllers
converged 10,000/10,000 times and neither exceeded 20 ms on the development
machine. Baseline mean/P99 time was 0.2346/0.3238 ms with at most 37 observed
iterations. The candidate mean/P99 was 0.0816/0.1229 ms with at most 13 observed
iterations. Host timings do not replace Raspberry Pi hardware timing tests.

## Verification and known baseline issues

The 47 TinyMPC, augmented-integral, state-machine regression, and iteration
harness tests pass. The project-wide self-check still reports four pre-existing
issues from the supplied archive:

1. the legacy supervisor test expects a 60.01-second request to fail, while the
   current Figure-8 implementation intentionally supports longer session caps;
2. one legacy YAML fixture lacks the current `ch7_policy` field;
3. two Stage 2 compatibility tests require an external
   `sitl_outer_loop_stage`/`sitl_guided_stage2` tree not included in the archive.

These unrelated safety interfaces were not changed to make the suite appear
green.

## Required next gates

Before any flight-live use, keep the delivered candidate in audit-only mode and
complete, in order: Raspberry Pi 10,000-solve timing, prop-off bench, restrained
low-amplitude hover/step validation, and only then a separately authorized
Figure-8 flight. Offline results are evidence for a candidate, not flight proof.
