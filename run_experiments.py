from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from batch_evaluation import BatchEvaluator
from export_extractions_to_excel import COLUMNS, load_extractions
from models import get_client
from data_models import ExtractionResult
from prompting import build_prompt
from spinner import Spinner
from utils import (
    get_icos,
    list_pmcids,
    load_annotations,
)

load_dotenv(find_dotenv())

def run_experiments(
    model: str,
    pdf_folder: str,
    gold_path: str,
    prompt_template: str,
    output_root: str,
    pmcids: Optional[List[int]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    fewshot_pdf_paths: Optional[List[str]] = None,
    fewshot_prompt_template: Optional[str] = None,
    prompt_label: Optional[str] = None,
) -> None:

    # 1. Setup paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = prompt_label or Path(prompt_template).stem
    run_name = f"{model.replace('-', '_')}_{label}_{ts}"
    run_dir = Path(output_root) / run_name
    extractions_dir = run_dir / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting run: {run_name}")
    print(f"Output directory: {run_dir}")

    # 2. Load Data
    annotations = load_annotations(gold_path)
    all_pmcids = pmcids or list_pmcids(pdf_folder)
    if not all_pmcids:
        print(f"No PDFs found in {pdf_folder}")
        return

    # 3. Setup Client
    client = get_client(model, temperature=temperature)

    # 4. Processing Loop
    success_count = 0

    # Prepare few-shot config
    pdf_context_paths = []
    num_fewshot = 0
    active_prompt_template = prompt_template

    if fewshot_pdf_paths:
        pdf_context_paths.extend([str(Path(p)) for p in fewshot_pdf_paths])
        num_fewshot = len(fewshot_pdf_paths)
        if fewshot_prompt_template:
            active_prompt_template = fewshot_prompt_template

    for idx, pmcid in enumerate(all_pmcids, start=1):
        print(f"[{idx}/{len(all_pmcids)}] Processing PMCID={pmcid}...")

        # Check PDF
        pdf_path = Path(pdf_folder) / f"{pmcid}.pdf"
        if not pdf_path.exists():
            # Try alt name
            pdf_path = Path(pdf_folder) / f"PMCID{pmcid}.pdf"
            if not pdf_path.exists():
                print(f"  Warning: PDF not found for {pmcid}")
                continue

        # Build Prompt
        ico_rows = get_icos(pmcid, annotations=annotations)
        prompt = build_prompt(article_text=None, ico_rows=ico_rows, base_prompt_path=active_prompt_template)
        
        # Combine PDFs for this call
        current_pdf_paths = pdf_context_paths + [str(pdf_path)]

        # Generate
        try:
            with Spinner("  Extracting"):
                response_text = client.generate(
                    prompt=prompt,
                    pdf_paths=current_pdf_paths,
                    max_tokens=max_tokens,
                    num_fewshot_pdfs=num_fewshot
                )
        except Exception as e:
            print(f"  Error generating response: {e}")
            # Write error log
            (run_dir / "errors.log").open("a").write(f"{pmcid}: {e}\n")
            continue

        # Parse & Save
        result = ExtractionResult.from_response(
            pmcid=pmcid,
            response=response_text,
            model=model,
            prompt_strategy=label,
        )
        result.save(extractions_dir / f"{pmcid}_{label}.jsonl")
        success_count += 1

    # 5. Evaluate & Export
    print("\n--- Evaluation ---")
    if success_count > 0:
        evaluator = BatchEvaluator(gold_path=gold_path, output_dir=str(run_dir / "evaluation"))
        eval_results = evaluator.evaluate_directory(
            predictions_dir=str(extractions_dir),
            suffix_filter=f"_{label}",
            run_name=run_name
        )
        print(f"Micro F1: {eval_results.get('micro_f1', 'N/A')}")

        # Export CSV
        rows = load_extractions(extractions_dir, suffix=f"_{label}")
        if rows:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(run_dir / "results.csv", index=False)
            print(f"Results saved to {run_dir}/results.csv")
    else:
        print("No successful extractions to evaluate.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--pdf-folder", default="data/PDF_test")
    parser.add_argument("--gold-path", default="gold-standard/gold_standard_clean.json")
    parser.add_argument("--prompt-template", default="prompt_templates/guided_prompt_GPT5_direct.md")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--pmcids", nargs="+", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--fewshot-pdf-paths", nargs="+")
    parser.add_argument("--fewshot-prompt-template")
    parser.add_argument("--prompt-label")

    args = parser.parse_args()

    run_experiments(
        model=args.model,
        pdf_folder=args.pdf_folder,
        gold_path=args.gold_path,
        prompt_template=args.prompt_template,
        output_root=args.output_root,
        pmcids=args.pmcids,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        fewshot_pdf_paths=args.fewshot_pdf_paths,
        fewshot_prompt_template=args.fewshot_prompt_template,
        prompt_label=args.prompt_label
    )

if __name__ == "__main__":
    main()
