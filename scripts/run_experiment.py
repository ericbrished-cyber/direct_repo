import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.config import RESULTS_DIR
from src.utils.data_loader import DataLoader
from src.utils.parsing import clean_and_parse_json
from src.prompts.builder import PromptBuilder
from src.models.claude import ClaudeModel
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel

def main():
    parser = argparse.ArgumentParser(description="Run RCT Meta-Analysis Extraction")
    parser.add_argument("--model", type=str, required=True, choices=["claude", "gpt", "gemini"], help="Model to use")
    parser.add_argument("--strategy", type=str, default="zero-shot", choices=["zero-shot", "few-shot"], help="Prompting strategy")
    parser.add_argument("--split", type=str, default="DEV", help="Data split to run on (e.g., DEV, TEST)")
    args = parser.parse_args()

    # 1. Init & Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model}_{args.strategy}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting run: {run_name}")
    print(f"Output directory: {output_dir}")

    # 2. Select Model
    if args.model == "claude":
        model = ClaudeModel()
    elif args.model == "gpt":
        model = GPTModel()
    elif args.model == "gemini":
        model = GeminiModel()
    else:
        raise ValueError("Unknown model")

    # 3. Load Data
    loader = DataLoader()
    pmcids = loader.get_split_pmcids(args.split)
    print(f"Found {len(pmcids)} documents in split '{args.split}'")

    prompt_builder = PromptBuilder(loader)

    # Statistics
    stats = {
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
            payload = prompt_builder.build(pmcid, mode=args.strategy)

            # Generate
            # Note: This will raise NotImplementedError until user implements the API call
            raw_text, usage = model.generate(payload)

            # Parse
            parsed_json = clean_and_parse_json(raw_text)

            # Save Atomically
            result_file = output_dir / f"{pmcid}.json"
            result_data = {
                "pmcid": pmcid,
                "raw_text": raw_text,
                "extraction": parsed_json,
                "usage": usage,
                "model": args.model,
                "strategy": args.strategy
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

if __name__ == "__main__":
    main()
