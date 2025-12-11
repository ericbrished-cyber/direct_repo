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
        pmcids = {
            str(entry['pmcid'])
            for entry in self._data
            if entry.get('split') == split_name
        }
        return sorted(list(pmcids))

    def get_entry(self, pmcid: str) -> List[Dict]:
        """
        Returns the list of entries in gold standard for a given pmcid.
        """
        return [
            entry for entry in self._data
            if str(entry['pmcid']) == str(pmcid)
        ]

    def get_few_shot_examples(self) -> List[Dict[str, object]]:
        """
        Returns a list of few-shot examples with their PDF, PMCID, and gold JSON string.
        """
        few_shot_pmcids = self.get_split_pmcids("FEW-SHOT")

        wanted_keys = ["outcome", "intervention", "comparator",
                        "outcome_type", "intervention_events", "intervention_group_size", "comparator_events",
                        "comparator_group_size", "intervention_mean", "intervention_standard_deviation", "comparator_mean",
                        "comparator_standard_deviation"]

        examples = []

        for pmcid in few_shot_pmcids:
            pdf_path = self.get_pdf_path(pmcid)
            entry = self.get_entry(pmcid)
            filtered_entry = [
                {k: item[k] for k in wanted_keys if k in item}
                for item in entry
            ]

            gold_for_few_shot_str = json.dumps(filtered_entry)

            examples.append({
                "pmcid": pmcid,
                "pdf_path": pdf_path,
                "answer": gold_for_few_shot_str,
            })

        return examples

    def get_icos(self, pmcid: str) -> List[Dict[str, str]]:
        """
        Returns [{"outcome" = x, "intervention" = y, "comparator" = z, "outcome_type" = x}] for a given pmcid.
        This for all targeted ICO in PMCID article
        """
        wanted_keys = ["outcome", "intervention", "comparator",
                        "outcome_type"]
        entry = self.get_entry(pmcid)
        return [
                {k: item[k] for k in wanted_keys if k in item}
                for item in entry
            ]

    def get_pdf_path(self, pmcid: str) -> Path:
        """
        Returns the path to the PDF file for a given PMCID.
        """
        pdf_path = self.pdf_dir / f"{pmcid}.pdf"
        return pdf_path
