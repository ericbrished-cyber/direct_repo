# LLM RCT Extraction & Evaluation

Code for my Bachelor's thesis on extracting arm-level statistics from randomized controlled trial PDFs with multimodal LLMs. Thesis link: pending.

## Abstract
Extracting numerical data from RCT reports is time-consuming and error-prone. While LLMs show potential on this task, previous work concluded that models are not reliable enough for research synthesis without human oversight. We tasked three multimodal models (GPT-5.2, Gemini 3 Pro, and Claude Haiku 4.5) with extracting arm-level statistics (including data from figures) from full-text PDFs. Specifically, we evaluated whether providing full-text PDFs as few-shot examples improved performance compared to a zero-shot baseline. GPT-5.2 achieved the highest zero-shot performance (F1: 85.5%, Exact Match: 58.3%), though results were comparable across models (F1 range: 82.9%-85.5%, Exact Match range: 51.9%-58.3%). The inclusion of full-text few-shot examples diminished the performance of all models compared to the zero-shot baseline: by -0.1% in F1-score for Claude Haiku, by -1.1% for GPT, and by -0.6% for Gemini. The 95% paired cluster bootstrap intervals effectively ruled out the possibility of any substantial improvement, likely attributable to large distracting context windows. We further conducted a thorough qualitative discrepancy analysis, identifying that a large portion of discrepancies stemmed from external factors (e.g., errors in the gold standard, ambiguous reporting, etc.) rather than intrinsic model errors. Consequently, we consider our reported metrics to be conservative estimates of true performance and conclude that models are likely more fit for the task of assisting meta-analytic work than previously assumed.

## Main functionality
- Build ICO-specific prompts from the gold standard and run zero-shot or few-shot extraction on full-text PDFs.
- Support GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, and Claude Haiku 4.5 for extraction.
- Evaluate against the gold standard with precision/recall/F1, RMSE, exact match, and paired cluster bootstrap CIs; save per-field details for analysis.

## Data layout
- `data/pdfs/`: RCT PDFs named by PMCID.
- `data/gold_standard_clean.json`: gold standard labels with DEV/TEST/FEW-SHOT splits.
- `data/results/<timestamp>_<model>_<strategy>_<split>/`: run outputs.
- `data/debug/`: prompt payloads for `--dry-run`.

## Setup
1. `pip install -r requirements.txt`
2. Create a `.env` file (or export in your shell):
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`

## Usage

### Run full pipeline (extraction + evaluation)
```bash
python scripts/run_experiment.py --model gpt --strategy zero-shot --split DEV
```

Options:
- `--model`: `gpt`, `claude`, `claude-haiku`, `gemini`
- `--strategy`: `zero-shot`, `few-shot`
- `--split`: `DEV`, `TEST`
- `--pmcid`: run a single PMCID
- `--skip-eval`: extraction only

### Extraction only (more control)
```bash
python scripts/run_extraction.py --model gpt --strategy zero-shot --split DEV
```

Options:
- `--pmcid`: run a single PMCID
- `--dry-run`: dump prompt payloads to `data/debug/` without API calls

### Evaluation only
```bash
python scripts/run_evaluation.py --run_folder <folder_name_from_results> --split DEV
```

## Output
Results are saved in `data/results/<timestamp>_<model>_<strategy>_<split>/`:
- `*.json`: extraction output per paper
- `evaluation_metrics.json`: aggregated metrics + bootstrap CIs
- `evaluation_details.csv`: per-field detailed results for analysis
- `run_metadata.json`: run summary and retry stats
- `*_error.txt`: per-PMCID failures (if any)
