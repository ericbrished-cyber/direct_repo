import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
from src.utils.data_loader import DataLoader
from src.prompts.templates import SYSTEM_PROMPT

@dataclass
class PromptPayload:
    instruction: str
    target_pdf: Path
    # Few-shot examples carry the PDF, their own ICO-specific instruction, and the answer string
    few_shot_examples: List[Dict[str, Any]]
    target_icos: List[Dict] # The specific targets to look for

class PromptBuilder:
    """
    Responsibility: Construct the PromptPayload object.
    """
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def _build_instruction(self, icos: List[Dict]) -> str:
        """Create an instruction string for a given ICO list."""
        ico_list_lines = []
        for idx, ico in enumerate(icos, start=1):
            ico_list_lines.append(
                f"- ICO {idx}:\n"
                f"    outcome: {ico.get('outcome')}\n"
                f"    intervention: {ico.get('intervention')}\n"
                f"    comparator: {ico.get('comparator')}\n"
                f"    outcome_type: {ico.get('outcome_type')}"
            )
        ico_list_str = "\n".join(ico_list_lines)

        return SYSTEM_PROMPT.replace("{ico_list}", ico_list_str)

    def build(self, target_pmcid: str, mode: str = "zero-shot") -> PromptPayload:
        """
        Accepts target_pmcid and mode ("zero-shot" or "few-shot").
        Fetches the Target PDF and ICOs (the specific outcomes to extract).
        If mode == "few-shot", gathers few-shot examples with their own ICO-specific instructions.
        Returns a PromptPayload.
        """
        target_pdf = self.loader.get_pdf_path(target_pmcid)
        target_icos = self.loader.get_icos(target_pmcid)
        instruction = self._build_instruction(target_icos)

        few_shot_examples = []
        if mode == "few-shot":
            for example in self.loader.get_few_shot_examples():
                example_icos = self.loader.get_icos(example["pmcid"])
                example_instruction = self._build_instruction(example_icos)
                few_shot_examples.append({
                    "pdf_path": example["pdf_path"],
                    "instruction": example_instruction,
                    "answer": example["answer"],
                })

        return PromptPayload(
            instruction=instruction,
            target_pdf=target_pdf,
            few_shot_examples=few_shot_examples,
            target_icos=target_icos
        )
