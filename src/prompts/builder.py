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

        # We construct a specific instruction for this PDF, listing the targets.
        # This is appended to the generic system prompt or passed as user message.
        # For now, let's keep the generic system prompt in the payload or we can combine.
        # The requirement says "PromptPayload(instruction, target_pdf, few_shot_examples)"

        # Let's customize the instruction to include the targets
        targets_str = "\n".join([f"- Outcome: {ico.get('outcome', 'Unknown')} (Intervention: {ico.get('intervention')}, Comparator: {ico.get('comparator')})" for ico in target_icos])

        instruction = f"{SYSTEM_PROMPT}\n\nPlease extract data for the following specific outcomes:\n{targets_str}\n"

        few_shot_examples = []
        if mode == "few-shot":
            few_shot_examples = self.loader.get_few_shot_examples()

        return PromptPayload(
            instruction=instruction,
            target_pdf=target_pdf,
            few_shot_examples=few_shot_examples,
            target_icos=target_icos
        )
