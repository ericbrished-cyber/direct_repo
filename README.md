# LLM_RCTs_Extract
Direct LLM-based extraction of statistical results from RCT PDFs (LangExtract removed).

## End-to-end flow
- `run_experiments.py` loads config (model, PMCIDs, prompt strategy), gold annotations, and ICO schema.
- PDFs from `data/PDF_test` are converted to markdown on-demand.
- Prompts are built as zero-shot or few-shot (schema + optional examples + article text).
- LLM call → JSON extraction → validation (`models.ExtractionResult`).
- Always evaluate vs gold (precision/recall/F1) and export Excel/CSV to `outputs/{run}`.

## Quick start
```bash
python run_experiments.py \
  --model gpt-5.1-mini \
  --prompt-strategy few-shot \
  --pdf-folder data/PDF_test \
  --markdown-folder data/Markdown
```
- Outputs live in `outputs/<run_name>/` (auto-named if not provided).
- Results are saved as JSONL under `extractions/`, metrics under `evaluation/`, and tabular exports as `results.xlsx` and `results.csv`.

## Using the helper wrapper
`run_task.py` shows a minimal programmatic entrypoint:
```bash
python run_task.py
```
Tune the `ExperimentConfig` there if you prefer Python over CLI flags.

## Export predictions to Excel only
If you already have JSONL outputs in the new direct schema, convert them to a spreadsheet:
```bash
python export_extractions_to_excel.py \
  --input outputs/my_run/extractions \
  --output outputs/my_run/results.xlsx \
  --suffix _few-shot
```
