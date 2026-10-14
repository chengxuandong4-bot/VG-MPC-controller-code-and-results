# CSTR × DeepSeek-v4-flash: five-repetition summary

All 15 formal runs completed with 200 candidates each (3,000 candidates total).

## Formal best score by repetition

| Rep | Single | Dual-objective | Dual-strategy |
|---:|---:|---:|---:|
| 1 | 0.242948 | 0.287662 | 0.283640 |
| 2 | 0.250891 | 0.273602 | 0.264541 |
| 3 | 0.256086 | 0.247558 | 0.241786 |
| 4 | 0.279155 | 0.289905 | 0.293561 |
| 5 | 0.263252 | 0.284158 | 0.284719 |

## Descriptive statistics for formal best score

| Mode | Mean | SD | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| single | 0.258466 | 0.013737 | 0.256086 | 0.242948 | 0.279155 |
| dual_objective | 0.276577 | 0.017385 | 0.284158 | 0.247558 | 0.289905 |
| dual_strategy | 0.273650 | 0.020719 | 0.283640 | 0.241786 | 0.293561 |

## API and acceptance accounting

| Mode | API successes | Fallbacks | Accepted | Total |
|---|---:|---:|---:|---:|
| single | 995 | 5 | 999 | 1000 |
| dual_objective | 981 | 19 | 999 | 1000 |
| dual_strategy | 981 | 19 | 999 | 1000 |

## Provenance caveat

The frozen legacy protocol allows fallback candidates to update the incumbent. Therefore, the formal result and the best directly returned API candidate are both retained. See `aggregate_runs.csv` for per-run provenance and dual-strategy seed lineage.
