# LLM Extraction of RCT Statistics (Bachelor Thesis)

Codebase for the Bachelor's thesis: **"Extracting Arm-Level Statistics from Randomized Controlled Trials with LLMs: A Comparison of Zero-Shot and Few-Shot Prompting"**.

**Author:** Isak Truedson and Eric Brished
**Supervisors:** Måns Magnusson (UU) & Gustav Nilsonne (KI)
**Thesis Link:** [PUT_LINK_HERE]

**Project Status:** This repository contains experimental research code developed during Spring 2026. It is designed for reproducibility and validation of the thesis methodology, not as production-ready software.

## Project Overview
This pipeline automates the extraction of statistical data from full-text RCT PDFs. It uses multimodal Large Language Models to extract arm-level statistics (means, standard deviations, sample sizes) and evaluates performance against a gold standard.

**Main Functionality:**
* Builds prompts dynamically based on gold-standard data (Intervention, Comparator, Outcome).
* Supports Zero-Shot and Few-Shot extraction strategies.
* Supports GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, and Claude Haiku 4.5.
* Evaluates results using Precision/Recall/F1, RMSE, and Exact Match with Paired Cluster Bootstrap Confidence Intervals.

## Data Layout
* `data/pdfs/`: Raw RCT PDFs (named by PMCID).
* `data/gold_standard_clean.json`: Ground truth labels with DEV/TEST/FEW-SHOT splits.
* `data/results/`: Outputs organized by run folder (timestamped).
* `data/debug/`: JSON payloads generated during dry-runs.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```
## Usage

### 1. Run Full Pipeline (Extraction + Evaluation)

Runs extraction on the PDFs and immediately calculates metrics.

```bash
python scripts/run_experiment.py --model gpt --strategy zero-shot --split DEV
```

**Arguments:**

* `--model`: `gpt`, `claude`, `claude-haiku`, `gemini`
* `--strategy`: `zero-shot`, `few-shot`
* `--split`: `DEV`, `TEST`
* `--pmcid`: (Optional) Run a single paper by ID.
* `--skip-eval`: Run extraction only.

### 2. Extraction Only

Useful for debugging or generating data without scoring.

```bash
python scripts/run_extraction.py --model gpt --strategy zero-shot --split DEV --dry-run

```

**Options:**

* `--dry-run`: Saves prompt payloads to `data/debug/` without calling the API.

### 3. Evaluation Only

Recalculate metrics for an existing run.

```bash
python scripts/run_evaluation.py --run_folder <folder_name> --split DEV
```

## Output

Results are saved in `data/results/<timestamp>_<model>_<strategy>_<split>/`:

* `*.json`: Extraction output per paper.
* `evaluation_metrics.json`: Aggregated metrics and confidence intervals.
* `evaluation_details.csv`: Detailed results per field (for analysis).
* `run_metadata.json`: Run configuration and stats.
* `*_error.txt`: Log of any failed extractions.
