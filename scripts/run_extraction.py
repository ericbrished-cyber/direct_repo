import json
import argparse
import sys
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Add project root to path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR
from src.utils.data_loader import DataLoader
from src.utils.WIP_parsing import clean_and_parse_json 
from src.prompts.builder import PromptBuilder
from src.models.gpt import GPTModel
from src.models.claude import ClaudeModel
from src.models.gemini import GeminiModel

def run_extraction(model_name: str, strategy: str, split: str, temperature: float = 0.0, pmcids = None):
    # 1. Setup Run Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_{strategy}_{split}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Extraction Run: {run_name} ---")
    print(f"Output Dir: {output_dir}")

    # 2. Initialize Components
    loader = DataLoader()
    #Get all PMCIDs for a split or PMCID overwrite to just run a single article.
    pmcids = pmcids or loader.get_split_pmcids(split)
    prompt_builder = PromptBuilder(loader)
    
    # Model Factory
    if model_name == "gpt":
        model = GPTModel()
    elif model_name == "claude":
        model = ClaudeModel()
    elif model_name == "gemini":
        model = GeminiModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Found {len(pmcids)} documents in split '{split}'")

    # 3. Extraction Loop
    stats = {"total": len(pmcids), "successful": 0, "failed": 0, "empty": 0,
              "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"), "end_time": None,}
    
    for pmcid in tqdm(pmcids, desc="Extracting"):
        try:
            # Build Prompt
            payload = prompt_builder.build(pmcid, mode=strategy)

            # Generate
            raw_text, usage = model.generate(payload, temperature=temperature)

            # Parse
            parsed_data = clean_and_parse_json(raw_text)
            
            # --- DATA VALIDATION & INJECTION ---
            extraction_list = []
            
            if parsed_data:
                # Normalize to list
                if isinstance(parsed_data, dict):
                    # Handle wrapper keys like {"extractions": [...]}
                    if "extractions" in parsed_data and isinstance(parsed_data["extractions"], list):
                        raw_list = parsed_data["extractions"]
                    else:
                        raw_list = [parsed_data]
                elif isinstance(parsed_data, list):
                    raw_list = parsed_data
                else:
                    raw_list = []

                # Inject PMCID into every extracted object
                for item in raw_list:
                    if isinstance(item, dict):
                        item['pmcid'] = str(pmcid)  # <--- CRITICAL INJECTION
                        extraction_list.append(item)
            
            if not extraction_list:
                stats["empty"] += 1

            # Save Individual Result (Atomic Save)
            result_file = output_dir / f"{pmcid}.json"
            file_data = {
                "pmcid": pmcid,
                "config": {"model": model_name, "strategy": strategy, "temp": temperature},
                "usage": usage,
                "raw_text": raw_text,
                "extraction": extraction_list 
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=2)

            stats["successful"] += 1

        except Exception as e:
            print(f"Error processing {pmcid}: {e}")
            stats["failed"] += 1
            # Save error log to avoid crashing the whole run
            with open(output_dir / f"{pmcid}_error.txt", 'w') as f:
                f.write(str(e))

    # 4. Save Run Metadata
    stats["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nExtraction complete. Results saved to {output_dir}")
    print(f"Run 'python scripts/run_evaluation.py --run_folder {run_name} --split {split}' to evaluate.")

    return run_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extraction Phase")
    parser.add_argument("--model", type=str, required=True, choices=["gpt", "claude", "gemini"])
    parser.add_argument("--strategy", type=str, default="zero-shot", choices=["zero-shot", "few-shot"])
    parser.add_argument("--split", type=str, default="DEV", help="Split to extract (DEV, TEST)")
    parser.add_argument("--pmcid", help="Run only this PMCID")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run_extraction(args.model, args.strategy, args.split, args.temperature, pmcids=[args.pmcid] if args.pmcid else None)