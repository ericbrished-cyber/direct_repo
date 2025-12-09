import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = DATA_DIR / "results"
GOLD_STANDARD_PATH = DATA_DIR / "gold_standard.json"

# Model Versions (Using placeholders/defaults as requested, can be overridden via env vars)
CLAUDE_MODEL_VERSION = os.getenv("CLAUDE_MODEL_VERSION", "claude-3-opus-20240229")
GPT_MODEL_VERSION = os.getenv("GPT_MODEL_VERSION", "gpt-4-turbo")
GEMINI_MODEL_VERSION = os.getenv("GEMINI_MODEL_VERSION", "gemini-1.5-pro")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
