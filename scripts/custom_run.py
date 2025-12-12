import sys
import os

# Lägg till föräldramappen (roten) i Pythons sökväg
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_extraction import run_extraction
from scripts.run_evaluation import run_evaluation_task

# ==========================================
# CUSTOM CONFIGURATION
# ==========================================

# Select Model: "claude", "gpt", or "gemini"
MODEL = "gemini"

# Select Split: "DEV", "TRAIN", "TEST"
SPLIT = "DEV"

# Select Strategy: "zero-shot" or "few-shot"
STRATEGY = "zero-shot"

# Select Temperature: 0.0 (deterministic) to 1.0 (creative)
#TEMPERATURE = 0.0

# Optional: run a specific list of PMCIDs (strings or ints). Leave empty to run the whole split.
PMCIDS = [
    # "2681019", "3169777",
]

if __name__ == "__main__":
    run_name = run_extraction(
        model_name=MODEL,
        strategy=STRATEGY,
        split=SPLIT,
        pmcids=PMCIDS or None,
        dry_run=False,
    )
    run_evaluation_task(run_folder=run_name, split=SPLIT)
