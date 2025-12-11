import argparse
import sys
import os

# Add the project root to Python path so we can import from 'scripts' and 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_extraction import run_extraction
from scripts.run_evaluation import run_evaluation_task

def main():
    parser = argparse.ArgumentParser(description="Run full RCT Experiment (Extraction + Evaluation)")
    
    # Model Args
    parser.add_argument("--model", type=str, required=True, choices=["claude", "gpt", "gemini"], help="Model to use")
    parser.add_argument("--strategy", type=str, default="zero-shot", choices=["zero-shot", "few-shot"], help="Prompting strategy")
    parser.add_argument("--split", type=str, default="DEV", help="Data split to run on (e.g., DEV, TEST)")
    parser.add_argument("--pmcid", help="Run only this PMCID")    
    
    # Flags
    parser.add_argument("--skip-eval", action="store_true", help="If set, only runs extraction without evaluation")

    args = parser.parse_args()

    print("-" * 60)
    print("Starting Experiment Pipeline")
    print("-" * 60)

    # ---------------------------------------------------------
    # STEP 1: EXTRACTION
    # ---------------------------------------------------------
    if args.pmcid:
        print(f"\nPhase 1: Running Extraction ({args.model}, pmcid={args.pmcid})...")
    else:
        print(f"\nPhase 1: Running Extraction ({args.model}, split={args.split})...")
    
    # Call the extraction script. 
    # It returns the 'run_name' (e.g., "20251211_gpt_zero-shot_DEV")
    run_name = run_extraction(
        model_name=args.model,
        strategy=args.strategy,
        split=args.split,
        pmcids=[args.pmcid] if args.pmcid else None
    )
    
    if not run_name:
        print("Error: Extraction failed or returned no run name. Aborting.")
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: EVALUATION
    # ---------------------------------------------------------
    if not args.skip_eval:
        print(f"\nPhase 2: Evaluating run '{run_name}'...")
        
        run_evaluation_task(
            run_folder=run_name,
            split=args.split
        )
    else:
        print(f"\nPhase 2: Evaluation skipped. Results available in: data/results/{run_name}")

    print("\nExperiment pipeline completed successfully.")

if __name__ == "__main__":
    main()
