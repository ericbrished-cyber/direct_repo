import sys
import os

# Lägg till föräldramappen (roten) i Pythons sökväg
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.runner import run_evaluation

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
TEMPERATURE = 0.0


if __name__ == "__main__":
    run_evaluation(
        model_name=MODEL,
        strategy=STRATEGY,
        split=SPLIT,
        temperature=TEMPERATURE
    )
