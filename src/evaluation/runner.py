import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from src.config import RESULTS_DIR, GOLD_STANDARD_PATH
from src.utils.data_loader import DataLoader
from src.utils.parsing import clean_and_parse_json
from src.prompts.builder import PromptBuilder
from src.models.claude import ClaudeModel
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel
from src.evaluation.metrics import calculate_metrics

def run_evaluation(model_name: str, strategy: str, split: str, temperature: float = 0.0):
    """
    Executes the evaluation loop for a given configuration.
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

    # Statistics & Data Collection
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
        "start_time": timestamp
    }
    
    all_predictions = [] # Samlar alla extraherade data här

    # 4. Loop
    for pmcid in tqdm(pmcids, desc="Processing PDFs"):
        try:
            # Build Payload
            payload = prompt_builder.build(pmcid, mode=strategy)

            # Generate
            raw_text, usage = model.generate(payload, temperature=temperature)

            # Parse
            parsed_json = clean_and_parse_json(raw_text)

            # Collect predictions for final evaluation
            if parsed_json:
                if isinstance(parsed_json, list):
                    all_predictions.extend(parsed_json)
                elif isinstance(parsed_json, dict):
                    all_predictions.append(parsed_json)

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

        except NotImplementedError as e:
            print(f"Skipping {pmcid}: {e}")
            break 
        except Exception as e:
            print(f"Error processing {pmcid}: {e}")
            stats["failed"] += 1
            error_file = output_dir / f"{pmcid}_error.txt"
            with open(error_file, 'w') as f:
                f.write(str(e))

    # 5. Automatic Evaluation & Reporting
    print("\n" + "="*40)
    print("PROCESSING COMPLETE. RUNNING EVALUATION...")
    print("="*40)

    # Save compiled results (så du slipper köra compile_results.py)
    final_results_file = output_dir / "final_results.json"
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2)
    
    # Load Gold Standard for this split
    try:
        with open(GOLD_STANDARD_PATH, 'r', encoding='utf-8') as f:
            full_gold = json.load(f)
        
        # Filtrera ut rätt split (DEV/TEST)
        gold_standard = [item for item in full_gold if item.get("split") == split]
        
        if not gold_standard:
            print(f"Warning: No gold standard data found for split '{split}'. Cannot evaluate.")
        else:
            # Calculate metrics
            metrics = calculate_metrics(all_predictions, gold_standard)
            
            # Save metrics to file
            with open(output_dir / "evaluation_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

            # Print Report to Console
            print(f"\nRESULTS FOR SPLIT: {split}")
            print("-" * 30)
            print(f"Precision:      {metrics['precision']:.2%}")
            print(f"Recall:         {metrics['recall']:.2%}")
            print(f"F1 Score:       {metrics['f1']:.2%}")
            print("-" * 30)
            print(f"True Positives: {metrics['true_positives']}")
            print(f"False Positives:{metrics['false_positives']}")
            print(f"False Negatives:{metrics['false_negatives']}")
            print("-" * 30)
            print(f"Full results saved to: {output_dir}")

    except Exception as e:
        print(f"Evaluation failed: {e}")

    # 6. Final Report Stats
    stats["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / "summary_report.json", 'w') as f:
        json.dump(stats, f, indent=2)