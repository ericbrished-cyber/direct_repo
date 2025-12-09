#!/bin/bash

# ==========================================
# Configuration
# ==========================================
# Change these variables to switch configurations
MODEL="claude"        # Options: claude, gpt, gemini
STRATEGY="zero-shot"  # Options: zero-shot, few-shot
SPLIT="DEV"           # Options: DEV, TEST

# ==========================================
# Execution
# ==========================================

# Ensure Python can find the src module
export PYTHONPATH=$PYTHONPATH:.

echo "----------------------------------------------------------------"
echo "Starting RCT Meta-Analysis Extraction"
echo "Model:    $MODEL"
echo "Strategy: $STRATEGY"
echo "Split:    $SPLIT"
echo "----------------------------------------------------------------"

python3 scripts/run_experiment.py --model "$MODEL" --strategy "$STRATEGY" --split "$SPLIT"
