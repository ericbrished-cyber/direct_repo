from prompting import build_prompt
from utils import load_annotations, get_icos

pmcid = 4987981
annotations = load_annotations("gold-standard/gold_standard_clean.json")
ico_rows = get_icos(pmcid, annotations=annotations)

prompt = build_prompt(
    article_text=None,  # eller sträng med text om du har den
    ico_rows=ico_rows,
    base_prompt_path="/Users/ericbrished/Desktop/direct_repository/prompt_templates/3llm_prompt_fewshot.md",
)
print(prompt)
