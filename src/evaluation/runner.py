import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from src.config import RESULTS_DIR
from src.utils.data_loader import DataLoader
from src.utils.parsing import clean_and_parse_json
from src.prompts.builder import PromptBuilder
from src.models.claude import ClaudeModel
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel

def run_evaluation(model_name: str, strategy: str, split: str, temperature: float = 0.0):
    """
    Executes the evaluation loop for a given configuration.

    Args:
        model_name: The name of the model to use ("claude", "gpt", "gemini").
        strategy: The prompting strategy ("zero-shot", "few-shot").
        split: The data split to use (e.g., "DEV", "TEST").
        temperature: The sampling temperature for the model (0.0 to 1.0).
    """

    # 1. Init & Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_{strategy}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting run: {run_name}")
    print(f"Configuration: Model={model_name}, Strategy={strategy}, Split={split}, Temp={temperature}")
    print(f"Output directory: {output_dir}")

    # 2. Select Model
    if model_name == "claude":
        model = ClaudeModel()
    elif model_name == "gpt":
        model = GPTModel()
    elif model_name == "gemini":
        model = GeminiModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 3. Load Data
    loader = DataLoader()
    pmcids = loader.get_split_pmcids(split)
    print(f"Found {len(pmcids)} documents in split '{split}'")

    prompt_builder = PromptBuilder(loader)

    # Statistics
    stats = {
        "config": {
            "model": model_name,
            "strategy": strategy,
            "split": split,
            "temperature": temperature
        },
        "total": len(pmcids),
        "successful": 0,
        "failed": 0,
        "total_tokens": 0,
        "start_time": timestamp
    }

    # 4. Loop
    for pmcid in tqdm(pmcids, desc="Processing PDFs"):
        try:
            # Build Payload
            payload = prompt_builder.build(pmcid, mode=strategy)

            # Generate
            raw_text, usage = model.generate(payload, temperature=temperature)

            # Parse
            parsed_json = clean_and_parse_json(raw_text)

            # Save Atomically
            result_file = output_dir / f"{pmcid}.json"
            result_data = {
                "pmcid": pmcid,
                "raw_text": raw_text,
                "extraction": parsed_json,
                "usage": usage,
                "model": model_name,
                "strategy": strategy,
                "temperature": temperature
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)

            stats["successful"] += 1
            # Update token stats if usage dict has standard keys
            # stats["total_tokens"] += usage.get("total_tokens", 0)

        except NotImplementedError as e:
            print(f"Skipping {pmcid}: {e}")
            break # Stop loop if API not implemented
        except Exception as e:
            print(f"Error processing {pmcid}: {e}")
            stats["failed"] += 1
            # Optionally save error log
            error_file = output_dir / f"{pmcid}_error.txt"
            with open(error_file, 'w') as f:
                f.write(str(e))

    # 5. Report
    stats["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / "summary_report.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("Run completed.")
