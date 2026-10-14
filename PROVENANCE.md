# Provenance notes

## Original campaign

`paper_results/original_campaign_manifest.csv` maps the 36 original backend--benchmark--mode conditions to their retained `summary_metrics.json` files and best-incumbent snapshots.

- 33 conditions have an exactly matched snapshot at the reported best generation and score.
- DeepSeek AUV dual-objective and dual-strategy both report `best_gen = 0` and `seed_q = 0.40241204256902879`. No generation snapshot with that seed incumbent is retained in the corresponding run directories; the release therefore includes the summaries but does not invent a controller snapshot.
- GPT-5.2 AUV dual-objective reports `best_q = 0.40260239232498263`, `best_source = human_guided_B030_FF35`, and a hybrid human-guided protocol in its retained summary. No matching generation snapshot is retained. The current supplementary text describes a discarded preliminary run followed by a restart without manually supplied controller parameters or candidates. Whether that description and the retained `human_guided_B030_FF35` endpoint refer to different runs remains to be established from the original records. Both statements are preserved here for author reconciliation; the endpoint is not relabeled as purely API-generated.

## Reported versus retained baseline fields

`paper_results/paper_reported_scores.csv` records the rounded values printed in the manuscript. Individual retained summaries sometimes contain slightly different baseline fields from the common rounded values printed in the paper tables. Both records are preserved rather than silently rewritten. The normalization and reporting lineage should be audited before a final tagged release.

## Supporting campaigns

- The five-block CSTR package contains the frozen protocol, aggregate results, and analysis scripts but omits the approximately 923 MB of full candidate histories.
- The controlled CSTR package contains the formal protocol/analysis code and aggregate 10-block result files but omits non-public pilots, authentication probes, raw block archives, and intermediate execution caches.
- The 10-block UAV values were recovered to four decimal places from the retained vector figure, as disclosed in the manuscript. They are not represented as original run-level logs.

## Hardware case

The hardware directory contains curated current code and selected configurations. Raw flight logs and motion-capture files are excluded from this lightweight code package. The manuscript treats the experiment as a single deployment-case comparison rather than a repeated hardware trial.

The repository retains the `rpi_pixhawk_stage4` package name needed by internal Python imports. Some adapter paths also expect `sitl_outer_loop_stage/python_controller/controller_interface_defaults.json`, which is not present in this release. Hardware offline tests have not been certified as passing from this repository tree.

## Security handling

Historical generation snapshots contained provider configuration files with secret-shaped values. All snapshot-level `llm_config.json` files are excluded. The retained workflow contains only `llm_config.example.json`, which reads credentials from the `LLM_API_KEY` environment variable.
