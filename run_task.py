from pathlib import Path

from run_experiments import ExperimentConfig, run_experiments

# Configuration
USE_FEWSHOT = False  # Set to False for zero-shot

def main():
    root = Path(__file__).resolve().parent
    
    # Base configuration
    base_config = {
        "model": "gpt-5.1",
        "pdf_folder": "data/PDF_dev",
        "gold_path": "gold-standard/gold_standard_clean.json",
        "temperature": 0.8,
        "max_tokens": 8000,
    }
    
    if USE_FEWSHOT:
        # Few-shot configuration
        config = ExperimentConfig(
            **base_config,
            prompt_label="fewshot",
            prompt_template="prompt_templates/3llm_prompt.md",
            fewshot_prompt_template="prompt_templates/pdf_fulltext_fewshot.md",
            fewshot_pdf_paths=[
                str(root / "few-shots/example1.pdf"),
                str(root / "few-shots/example2.pdf"),
            ],
        )
    else:
        # Zero-shot configuration
        config = ExperimentConfig(
            **base_config,
            prompt_label="zeroshot",
            prompt_template="prompt_templates/3llm_prompt.md",
            fewshot_prompt_template=None,
            fewshot_pdf_paths=None,
        )
    
    run_experiments(config)


if __name__ == "__main__":
    main()