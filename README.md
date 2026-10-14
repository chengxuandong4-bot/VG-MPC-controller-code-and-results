# VG-MPC: code and supporting artifacts

This repository accompanies *VG-MPC: Verification-guided generation and refinement of model predictive control code*. It contains the controller-generation workflow, benchmark and evaluation code, selected generated controllers, supporting analyses, and quadrotor deployment-case materials.

## Repository contents

| Path | Contents |
| --- | --- |
| `workflow/` | Single- and dual-generator refinement workflow and LLM configuration example. |
| `benchmarks/` | CSTR, fault-tolerant AUV, ground-vehicle, and quadrotor benchmarks. |
| `generated_controllers/original_campaign/` | Summaries for 36 backend–benchmark–mode conditions and 33 matched best-controller snapshots. |
| `supporting_campaigns/` | CSTR follow-up and controlled comparisons, plus the 10-block quadrotor analysis. |
| `hardware_quadrotor/` | Deployment-case code, selected configurations, tests, and design notes. |
| `paper_results/` | Paper scores, run manifest, and score-mapping check. |
| `environment/` | Python dependencies and MATLAB requirements. |
| `VG-MPC-generated-controller-artifacts.zip` | Original selected-artifact archive, indexed by `INDEX.csv`. |

`PROVENANCE.md` documents how retained artifacts relate to the reported results. `MANIFEST.csv` inventories the current repository files.

## Checks

Run these commands from the repository root:

```bash
python3 paper_results/validate_score_mapping.py
python3 build_release_manifest.py --check
```

The first checks the 36 retained best scores against the rounded paper table. The second verifies the current file inventory and checksums in `MANIFEST.csv` and `RELEASE_SHA256SUMS.txt`. Checksums for files inside the original ZIP are listed in `SHA256SUMS.txt`.

## Code entry points

- Refinement workflow: `workflow/run_unified_outer_loop.m`
- CSTR: `benchmarks/cstr/src/run_generation_experiment.m`
- AUV: `benchmarks/auv/src/evaluation/simulate_closed_loop.m`
- Ground vehicle: `benchmarks/ground_vehicle/src/run_rampc_python.py`
- Quadrotor simulation: `benchmarks/quadrotor/src/run_drone_sim.m`
- Paired quadrotor analysis: `supporting_campaigns/uav_ten_block_analysis/make_uav_ten_block_figure.py`
- Score mapping: `paper_results/validate_score_mapping.py`

Python dependencies are listed in `environment/requirements.txt`. The CSTR, AUV, and quadrotor benchmarks use MATLAB; CSTR also uses Optimization Toolbox. External LLM calls read an API key from the `LLM_API_KEY` environment variable. Hardware integration details are in `hardware_quadrotor/README.md`; vehicle operation requires platform-specific integration and safety review.

License: not specified.
