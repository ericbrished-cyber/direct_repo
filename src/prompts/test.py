import json
from src.utils.data_loader import DataLoader

from src.prompts.templates import SYSTEM_PROMPT


loader = DataLoader()

target_icos = loader.get_icos("4493951")

# Inject the current ICO list into the system prompt placeholder for transparency
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

print(instruction)
