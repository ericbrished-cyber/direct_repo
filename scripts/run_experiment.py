import argparse
from src.evaluation.runner import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Run RCT Meta-Analysis Extraction")
    parser.add_argument("--model", type=str, required=True, choices=["claude", "gpt", "gemini"], help="Model to use")
    parser.add_argument("--strategy", type=str, default="zero-shot", choices=["zero-shot", "few-shot"], help="Prompting strategy")
    parser.add_argument("--split", type=str, default="DEV", help="Data split to run on (e.g., DEV, TEST)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 to 1.0)")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        strategy=args.strategy,
        split=args.split,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
