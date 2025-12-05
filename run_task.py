"""
Convenience wrapper to trigger the new LangExtract-free pipeline.
Use `python run_experiments.py --help` for full CLI options.
"""

from run_experiments import ExperimentConfig, run_experiments

# model="claude-opus-4-5",
def main():
    config = ExperimentConfig(
        model="gemini-3.0-pro",
        prompt_label="direct", 
        pdf_folder="data/PDF_excel_test",
        markdown_folder=None,  # PDFs are sent directly to the model
        prompt_template="prompt_templates/guided_prompt_GPT5_direct.md",  # swap to your few-shot template when ready
    )
    run_experiments(config)


if __name__ == "__main__":
    main()
