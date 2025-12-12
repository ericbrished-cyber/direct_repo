import json
import argparse
import sys
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
import random

# Add project root to path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR
from src.utils.data_loader import DataLoader
from src.utils.WIP_parsing import clean_and_parse_json 
from src.prompts.builder import PromptBuilder
from src.models.gpt import GPTModel
from src.models.claude import ClaudeModel
from src.models.gemini import GeminiModel

# Configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
MAX_BACKOFF = 120
TOTAL_TIMEOUT_HOURS = 4

RETRYABLE_ERRORS = ("rate_limit", "overloaded", "timeout", "connection", "server_error")

def exponential_backoff(attempt: int) -> float:
    """Calculate wait time with exponential backoff and jitter."""
    wait = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
    jitter = random.uniform(0, wait * 0.1)
    return wait + jitter

def is_retryable_error(error: Exception) -> bool:
    """Check if error should trigger retry."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in RETRYABLE_ERRORS)

def get_failed_pmcids(output_dir: Path) -> set:
    """Get list of PMCIDs that have error files."""
    error_files = output_dir.glob("*_error.txt")
    return {f.stem.replace("_error", "") for f in error_files}

def extract_single_pdf(pmcid: str, model, prompt_builder, strategy: str, dry_run: bool = False):
    """Extract data from a single PDF. Returns (success: bool, data: dict, error: str)."""
    try:
        payload = prompt_builder.build(pmcid, mode=strategy)
        raw_text, usage = model.generate(payload, dry_run=dry_run)
        parsed_data = clean_and_parse_json(raw_text)
        
        extraction_list = []
        if parsed_data:
            if isinstance(parsed_data, dict):
                raw_list = parsed_data.get("extractions", [parsed_data])
            elif isinstance(parsed_data, list):
                raw_list = parsed_data
            else:
                raw_list = []

            for item in raw_list:
                if isinstance(item, dict):
                    item['pmcid'] = str(pmcid)
                    extraction_list.append(item)
        
        return True, {"extraction": extraction_list, "raw_text": raw_text, "usage": usage}, None

    except Exception as e:
        return False, None, str(e)

def save_result(pmcid: str, data: dict, output_dir: Path, model_name: str, strategy: str):
    """Save successful extraction result."""
    result_file = output_dir / f"{pmcid}.json"
    file_data = {
        "pmcid": pmcid,
        "config": {"model": model_name, "strategy": strategy},
        "usage": data.get("usage", {}),
        "raw_text": data.get("raw_text", ""),
        "extraction": data.get("extraction", [])
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(file_data, f, indent=2)

def save_error(pmcid: str, error: str, output_dir: Path):
    """Save error log."""
    error_file = output_dir / f"{pmcid}_error.txt"
    with open(error_file, 'w') as f:
        f.write(f"Error: {error}\n")

def run_extraction(model_name: str, strategy: str, split: str, 
                   pmcids=None, dry_run: bool = False):
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = "custom" if pmcids else split
    run_name = f"{timestamp}_{model_name}_{strategy}_{run_suffix}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Extraction: {run_name}")
    print(f"Output: {output_dir}")

    # Initialize
    loader = DataLoader()
    pmcids = pmcids or loader.get_split_pmcids(split)
    prompt_builder = PromptBuilder(loader)
    
    if model_name == "gpt":
        model = GPTModel()
    elif model_name == "claude":
        model = ClaudeModel()
    elif model_name == "gemini":
        model = GeminiModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Processing {len(pmcids)} documents")

    # Statistics
    stats = {
        "total": len(pmcids),
        "successful": 0,
        "failed": 0,
        "empty": 0,
        "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_retries": 0
    }
    
    retry_counts = {pmcid: 0 for pmcid in pmcids}
    start_time = datetime.now()
    timeout = timedelta(hours=TOTAL_TIMEOUT_HOURS)
    
    # Main loop
    pending_pmcids = list(pmcids)
    iteration = 0
    with tqdm(total=len(pmcids), desc="Processing") as pbar:
        while pending_pmcids:
            if datetime.now() - start_time > timeout:
                print(f"Timeout reached ({TOTAL_TIMEOUT_HOURS}h)")
                print(f"Remaining: {len(pending_pmcids)} PDFs")
                for pmcid in pending_pmcids:
                    save_error(pmcid, "TIMEOUT: extraction not completed within limit", output_dir)
                stats["failed"] += len(pending_pmcids)
                pbar.update(len(pending_pmcids))
                break

            iteration += 1
            print(f"\nIteration {iteration}: {len(pending_pmcids)} PDFs to process")

            for pmcid in list(pending_pmcids):
                attempt_number = retry_counts[pmcid] + 1

                if attempt_number > 1:
                    wait = exponential_backoff(attempt_number - 2)
                    print(f"Waiting {wait:.1f}s before attempt {attempt_number} for {pmcid}")
                    time.sleep(wait)
                    stats["total_retries"] += 1

                retry_counts[pmcid] += 1
                success, data, error = extract_single_pdf(pmcid, model, prompt_builder, strategy, dry_run)

                if success:
                    save_result(pmcid, data, output_dir, model_name, strategy)
                    error_file = output_dir / f"{pmcid}_error.txt"
                    if error_file.exists():
                        error_file.unlink()
                    stats["successful"] += 1
                    if not data["extraction"]:
                        stats["empty"] += 1
                    pending_pmcids.remove(pmcid)
                    pbar.update(1)
                    print(f"Success: {pmcid} (attempt {attempt_number})")
                    continue

                error_message = str(error)
                retryable = is_retryable_error(Exception(error_message))

                if not retryable:
                    print(f"Permanent error for {pmcid}")
                    save_error(pmcid, f"PERMANENT: {error_message}", output_dir)
                    stats["failed"] += 1
                    pending_pmcids.remove(pmcid)
                    pbar.update(1)
                    continue

                if retry_counts[pmcid] >= MAX_RETRIES:
                    print(f"Max retries reached for {pmcid}")
                    save_error(pmcid, f"MAX_RETRIES: {error_message}", output_dir)
                    stats["failed"] += 1
                    pending_pmcids.remove(pmcid)
                    pbar.update(1)
                else:
                    print(f"Retryable error for {pmcid} (attempt {attempt_number}/{MAX_RETRIES})")
                    save_error(pmcid, error_message, output_dir)

    # Save metadata
    stats["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats["final_failed"] = list(get_failed_pmcids(output_dir))
    
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total:       {stats['total']}")
    print(f"Successful:  {stats['successful']}")
    print(f"Failed:      {stats['failed']}")
    print(f"Empty:       {stats['empty']}")
    print(f"Retries:     {stats['total_retries']}")
    
    if stats["final_failed"]:
        print(f"\nFailed PDFs: {', '.join(stats['final_failed'])}")
    
    print(f"\nResults: {output_dir}")

    return run_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extraction Phase with Retry Logic")
    parser.add_argument("--model", type=str, required=True, choices=["gpt", "claude", "gemini"])
    parser.add_argument("--strategy", type=str, default="zero-shot", choices=["zero-shot", "few-shot"])
    parser.add_argument("--split", type=str, default="DEV", help="Split to extract (DEV, TEST)")
    parser.add_argument("--pmcid", help="Run only this PMCID")
    parser.add_argument("--dry-run", action="store_true", help="Build and dump prompts without calling the API")
    args = parser.parse_args()

    run_extraction(
        args.model,
        args.strategy,
        args.split,
        pmcids=[args.pmcid] if args.pmcid else None,
        dry_run=args.dry_run,
    )
