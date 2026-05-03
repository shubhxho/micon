# Speall MRI -- Sample Reports

Concrete examples of what the Speall MRI pipeline produces. Numbers in
`../Speall_MRI_Dataset_Info.json` describe the corpus as a whole; the files
in this folder show what one full study and every series in it look like.

## Source study

One GE SIGNA Pioneer 3.0T study acquired **2026-03-23** (`BRAIN + ANGIO CISS`
protocol), processed end-to-end through the modal pipeline. Full detail in
`series_index.json`. Schema is real; values are real.

| | |
|---|---|
| Study description | BRAIN + ANGIO CISS |
| Series in study | 19 (10 primary acquisitions + 9 derivative maps) |
| Scanner | GE SIGNA Pioneer 3.0T, 32-channel head coil |
| Software | PX26.1_R03_2128.b |
| Sex | M |

## Files at a glance

| Path | What it is |
|---|---|
| `series_index.json` | Index of all 19 series in the study (number, description, sequence type, file count, paths) |
| `study_summary.json` | Compact one-row-per-series summary (TR/TE, volume shape, quality grade) |
| `study_full_series_stats.json` | Full `series_stats.json` produced by the modal pipeline (~270 KB) -- every field for every series |
| `series/s{NNNN}_{label}.json` | Per-series `*_detail.json` for all 19 series in the study |
| `ai/s{NNNN}_{label}.json` | Per-series Gemma 4 annotation (see below) |
| `ai/study_ai_summary.json` | Aggregated AI summary across the whole study |
| `run_ai.py` | Re-runnable script that produced `ai/` (needs `OPENROUTER_API_KEY` in `../.env`) |

## `ai/` (internal-only enrichment)

Internal Gemma 4 enrichment over each series in this sample study.
Two prompts per primary series (annotation + tissue), one per derivative
map. Run in parallel via `run_ai.py`. Not for distribution.

```bash
cd /Users/shubh/Documents/micom
uv run --with openai --with python-dotenv python Speall_MRI_Samples/run_ai.py
```
