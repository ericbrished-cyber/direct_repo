import json
import argparse
from pathlib import Path
from src.config import RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description="Compile individual JSON results into one file.")
    parser.add_argument("--run_folder", type=str, required=True, help="Name of the run folder in data/results/")
    args = parser.parse_args()

    run_path = RESULTS_DIR / args.run_folder
    if not run_path.exists():
        print(f"Error: Directory {run_path} does not exist.")
        return

    all_results = []

    # Iterate over all JSON files that look like PMCxxxx.json (or just all .json files excluding report)
    for file_path in run_path.glob("*.json"):
        if file_path.name == "summary_report.json" or file_path.name == "final_results.json":
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # We want to extract the "extraction" part which should be a list of ICO objects
            # If the extraction failed or is None, we might skip or record empty
            extraction = data.get("extraction")

            if isinstance(extraction, list):
                all_results.extend(extraction)
            elif isinstance(extraction, dict):
                 # In case it wrapped a single object, though prompt asks for list
                 all_results.append(extraction)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Save final compiled result
    output_file = run_path / "final_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"Compiled {len(all_results)} entries into {output_file}")

if __name__ == "__main__":
    main()
