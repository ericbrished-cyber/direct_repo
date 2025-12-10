import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = DATA_DIR / "results"
# GOLD_STANDARD_PATH points to the correct location in the root gold-standard folder
GOLD_STANDARD_PATH = BASE_DIR / "gold-standard" / "gold_standard_clean.json"

# Model Versions
CLAUDE_MODEL_VERSION = os.getenv("CLAUDE_MODEL_VERSION", "claude-opus-4.5")
GPT_MODEL_VERSION = os.getenv("GPT_MODEL_VERSION", "gpt-5.1")
GEMINI_MODEL_VERSION = os.getenv("GEMINI_MODEL_VERSION", "gemini-2.5-pro")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
