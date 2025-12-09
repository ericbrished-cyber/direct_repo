import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.config import GOLD_STANDARD_PATH, PDF_DIR

class DataLoader:
    """
    Responsibility: Parse gold_standard.json and manage file paths.
    """
    def __init__(self, data_path: Path = GOLD_STANDARD_PATH, pdf_dir: Path = PDF_DIR):
        self.data_path = data_path
        self.pdf_dir = pdf_dir
        self._data = self._load_data()

    def _load_data(self) -> List[Dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Gold standard file not found at {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_split_pmcids(self, split_name: str) -> List[str]:
        """
        Returns a unique list of PMCIDs belonging to a split (e.g., "TEST", "DEV", "FEW-SHOT").
        """
        # Note: The split name in JSON might be case-sensitive. The user example uses "TEST".
        # We'll normalize to upper case for comparison if needed, but let's stick to strict matching first.
        # Some entries might not have a split defined (None), we filter those out.
        pmcids = {
            str(entry['pmcid'])
            for entry in self._data
            if entry.get('split') == split_name
        }
        return sorted(list(pmcids))

    def get_icos(self, pmcid: str) -> List[Dict]:
        """
        Returns the list of specific ICO targets (outcomes) for a given document.
        """
        return [
            entry for entry in self._data
            if str(entry['pmcid']) == str(pmcid)
        ]

    def get_few_shot_examples(self) -> List[Tuple[Path, str]]:
        """
        Returns (pdf_path, formatted_answer_str) for entries marked as split="FEW-SHOT".
        Returns a list of tuples.
        """
        # Since one PDF can have multiple entries, and few-shot usually implies
        # providing the PDF and the *full* expected output for that PDF.
        # However, the requirement says: "Returns (pdf_path, formatted_answer_str) for entries marked as split='FEW-SHOT'".
        # If a PDF has multiple outcomes, we should probably aggregate them into one JSON response string
        # or treat each ICO as a separate example?
        # The "atomic save" requirement suggests one JSON per document.
        # So likely we want to return the PDF and the JSON containing ALL ICOs for that PDF.

        few_shot_pmcids = self.get_split_pmcids("FEW-SHOT")
        examples = []

        for pmcid in few_shot_pmcids:
            pdf_path = self.get_pdf_path(pmcid)
            icos = self.get_icos(pmcid)
            # formatted_answer_str should be the JSON representation of the extraction
            # We assume the model is expected to output a list of these objects.
            formatted_answer_str = json.dumps(icos, indent=2)
            examples.append((pdf_path, formatted_answer_str))

        return examples

    def get_pdf_path(self, pmcid: str) -> Path:
        """
        Returns the path to the PDF file for a given PMCID.
        """
        # Assuming file naming is simply {pmcid}.pdf
        pdf_path = self.pdf_dir / f"{pmcid}.pdf"
        # We don't check existence here to keep it fast, or we could.
        # Since we use a symlink dir, the files should be there.
        return pdf_path
