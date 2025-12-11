Here is a clean, barebones `README.md` that captures the essential usage of your new pipeline structure.

````markdown
# LLM RCT Extraction & Evaluation

A pipeline for extracting statistical results for a given ICO (Intervention, Comparator, Outcome) from Randomized Controlled Trial (RCT) PDFs using LLMs (GPT, Claude, Gemini) and evaluating performance against a gold standard.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
````

2.  **Environment Variables**
    Create a `.env` file or set these in your terminal:
      - `OPENAI_API_KEY`
      - `ANTHROPIC_API_KEY`
      - `GOOGLE_API_KEY`

## Usage

### 1\. Run Full Experiment (Extraction + Evaluation)

Run the complete pipeline to extract data and immediately evaluate it.

```bash
python scripts/run_experiment.py --model gpt --strategy zero-shot --split DEV
```

  * **Options:**
      * `--model`: `gpt`, `claude`, `gemini`
      * `--strategy`: `zero-shot`, `few-shot`
      * `--split`: `DEV`, `TEST`

### 2\. Run Extraction Only

Extract data and save to `data/results/` without running metrics.

```bash
python scripts/run_experiment.py --model gpt --strategy zero-shot --skip-eval
```

### 3\. Run Evaluation Only

Re-run metrics on a previously saved extraction folder.

```bash
python scripts/run_evaluation.py --run_folder <folder_name_from_results> --split DEV
```

## Output

Results are saved in `data/results/<timestamp>_<config>/`:

  * `*.json`: Individual extraction files per paper.
  * `evaluation_metrics.json`: Final precision, recall, F1, and RMSE scores.
  * `run_metadata.json`: Logs and configuration details.

<!-- end list -->

```
```