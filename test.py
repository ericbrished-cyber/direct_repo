"""
Lightweight smoke checks for the LangExtract-free pipeline pieces.
"""

from prompting import build_prompt
from utils import get_icos, list_pmcids, load_annotations


def main():
    annotations = load_annotations()
    pmcids = list_pmcids("data/PDF_test")
    print(f"Gold rows: {len(annotations)}")
    print(f"PDFs discovered: {len(pmcids)}")

    if pmcids:
        pmcid = pmcids[0]
        ico_rows = get_icos(pmcid, annotations=annotations)
        prompt = build_prompt(
            article_text="Dummy article text.",
            ico_rows=ico_rows,
            base_prompt_path="prompt_templates/all_prompt_new.md",
            prompt_strategy="zero-shot",
        )
        print(f"Built prompt for {pmcid} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
