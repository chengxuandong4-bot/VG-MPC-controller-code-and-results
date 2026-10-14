# VG-MPC code and supporting artifacts

This repository accompanies *VG-MPC: Verification-guided generation and refinement of model predictive control code*. The original `VG-MPC-generated-controller-artifacts.zip`, `INDEX.csv`, and `SHA256SUMS.txt` are retained unchanged as the earlier selected-artifact archive. The directories below expand the inspectable code and supporting evidence; they do not retroactively change the scope of that archive.

The older `SHA256SUMS.txt` applies to the files in the extracted ZIP, including its historical README; it does **not** validate the expanded repository root. Use `RELEASE_SHA256SUMS.txt` or `python3 build_release_manifest.py --check` for the expanded tree.

## Contents

| Path | Material |
| --- | --- |
| `workflow/` | Retained single- and dual-generator outer-loop code and an environment-variable-based LLM configuration example. |
| `benchmarks/` | CSTR, fault-tolerant AUV, ground-vehicle, and quadrotor benchmark and evaluation code. |
| `generated_controllers/original_campaign/` | Summaries for the 36 original backend–benchmark–mode conditions and 33 exactly matched best-incumbent snapshots. |
| `supporting_campaigns/` | CSTR follow-up and controlled-comparison materials, plus the 10-block paired quadrotor analysis. |
| `hardware_quadrotor/` | Curated deployment-case source under its original `rpi_pixhawk_stage4` Python package name, selected configurations, tests, and design notes. |
| `paper_results/` | Rounded paper scores, retained-run manifest, and score-mapping check. |
| `environment/` | Python dependency list and MATLAB requirements. |
| `MANIFEST.csv`, `RELEASE_SHA256SUMS.txt`, `PROVENANCE.md` | Whole-repository file inventory, checksums, and result-provenance limitations. |

The three conditions without exactly matched best snapshots remain in the manifest. No missing configuration or run history has been reconstructed. See `PROVENANCE.md` before interpreting any condition as a directly API-generated controller.

## What can be checked

From the repository root, `python3 paper_results/validate_score_mapping.py` checks the 36 retained best scores against the rounded manuscript table. `python3 build_release_manifest.py --check` verifies the whole-repository inventory and checksums. The benchmark and workflow sources can be inspected; a clean end-to-end run of this assembled release tree has not yet been verified. The Python dependencies are listed in `environment/requirements.txt`; the MATLAB benchmarks (CSTR, AUV, and quadrotor) require MATLAB, and CSTR uses Optimization Toolbox. External LLM calls require user-supplied API credentials via environment variables. The package does not provide those credentials.

The retained package is **not** a complete, one-command reproduction of every reported result. Full candidate histories, raw hardware and motion-capture records, API metadata, videos, and some best-incumbent snapshots are not included. Stochastic LLM generations and external API behavior need not reproduce exactly. The 10-block quadrotor values were recovered to four decimal places from a retained vector figure, rather than original run-level logs. The original-campaign manifest and `PROVENANCE.md` give the boundaries for each evidence stream.

## Starting points

- Outer loop: `workflow/run_unified_outer_loop.m`
- CSTR: `benchmarks/cstr/src/run_generation_experiment.m`
- AUV: `benchmarks/auv/src/evaluation/simulate_closed_loop.m`
- Ground vehicle: `benchmarks/ground_vehicle/src/run_rampc_python.py`
- Quadrotor simulation: `benchmarks/quadrotor/src/run_drone_sim.m`
- Paired quadrotor analysis: `supporting_campaigns/uav_ten_block_analysis/make_uav_ten_block_figure.py`
- Reported-to-retained score check: `paper_results/validate_score_mapping.py`

The hardware source is provided for inspection and research comparison. Its original import namespace has been restored within `hardware_quadrotor/`, but an external interface-defaults file is not included; see `hardware_quadrotor/README.md`. Do not run it on a vehicle without platform-specific integration and safety review.

## Availability and interpretation

This is a source-and-artifact release with documented limits, not a claim that every experiment can be reproduced end to end from the retained files. The original-campaign provenance and baseline-field differences are described in `PROVENANCE.md`. The repository has no open-source license at present: the files are publicly viewable, but no reuse permission is granted by public visibility alone. Release-tag and citation metadata have not yet been assigned.
