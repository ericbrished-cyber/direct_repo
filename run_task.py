
from pathlib import Path

from run_experiments import ExperimentConfig, run_experiments

# model="claude-opus-4-5",
# model="gemini-3-pro-preview",
# model="gpt-5.1",
def main():
    root = Path(__file__).resolve().parent
    config = ExperimentConfig(
        model="gpt-5.1",
        prompt_label="fewshot",
        pdf_folder="data/PDF_excel_test",
        gold_path="gold-standard/gold_standard_clean.json",
        prompt_template="prompt_templates/3llm_prompt.md",
        fewshot_prompt_template="prompt_templates/pdf_fulltext_fewshot.md",
        fewshot_pdf_paths=[
            str(root / "few-shots/example1.pdf"),
            str(root / "few-shots/example2.pdf"),
        ],
        temperature=0.0,
        max_tokens=8000,
    )
    run_experiments(config)


if __name__ == "__main__":
    main()
