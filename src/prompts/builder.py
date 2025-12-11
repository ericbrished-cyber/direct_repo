import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from src.utils.data_loader import DataLoader
from src.prompts.templates import SYSTEM_PROMPT

@dataclass
class PromptPayload:
    instruction: str
    target_pdf: Path
    few_shot_examples: List[Tuple[Path, str]]  # List of (pdf_path, answer_string)
    target_icos: List[Dict] # The specific targets to look for

class PromptBuilder:
    """
    Responsibility: Construct the PromptPayload object.
    """
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def build(self, target_pmcid: str, mode: str = "zero-shot") -> PromptPayload:
        """
        Accepts target_pmcid and mode ("zero-shot" or "few-shot").
        Fetches the Target PDF.
        Fetches the Target ICOs (the specific outcomes to extract).
        If mode == "few-shot", calls loader.get_few_shot_examples() and appends them.
        Returns a dataclass PromptPayload(instruction, target_pdf, few_shot_examples).
        """
        target_pdf = self.loader.get_pdf_path(target_pmcid)
        target_icos = self.loader.get_icos(target_pmcid)

        # Inject the current target ICO list into the system prompt placeholder
        ico_list_lines = []
        for idx, ico in enumerate(target_icos, start=1):
            ico_list_lines.append(
                f"- ICO {idx}:\n"
                f"    outcome: {ico.get('outcome')}\n"
                f"    intervention: {ico.get('intervention')}\n"
                f"    comparator: {ico.get('comparator')}\n"
                f"    outcome_type: {ico.get('outcome_type')}"
            )
        ico_list_str = "\n".join(ico_list_lines)

        instruction = SYSTEM_PROMPT.replace("{ico_list}", ico_list_str)

        few_shot_examples = []
        if mode == "few-shot":
            few_shot_examples = self.loader.get_few_shot_examples()

        return PromptPayload(
            instruction=instruction,
            target_pdf=target_pdf,
            few_shot_examples=few_shot_examples,
            target_icos=target_icos
        )
