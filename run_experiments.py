from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from batch_evaluation import BatchEvaluator
from export_extractions_to_excel import COLUMNS, load_extractions
from llm_client import LLMClient
from models import ExtractionResult
from prompting import build_prompt
from spinner import Spinner
from utils import (
    get_icos,
    list_pmcids,
    load_annotations,
)

load_dotenv(find_dotenv())


@dataclass
class ExperimentConfig:
    model: str = "gpt-5.1"
    prompt_label: Optional[str] = None  # human-friendly name for the prompt/template
    pdf_folder: str = "data/PDF_test"
    gold_path: str = "gold-standard/gold_standard_clean.json"
    prompt_template: str = "prompt_templates/guided_prompt_GPT5_direct.md"
    output_root: str = "outputs"
    run_name: Optional[str] = None
    pmcids: Optional[List[int]] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    fewshot_pdf_paths: Optional[List[str]] = None
    fewshot_prompt_template: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        base = cls()
        data = asdict(base)
        for field in data:
            if hasattr(args, field) and getattr(args, field) is not None:
                data[field] = getattr(args, field)
        return cls(**data)


def _auto_run_name(config: ExperimentConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = config.model.replace("-", "_").replace(".", "_")
    label = config.prompt_label or Path(config.prompt_template).stem
    label_clean = label.replace("-", "_")
    return f"{model_clean}_{label_clean}_{ts}"


def _prepare_output_folders(config: ExperimentConfig) -> Dict[str, Path]:
    run_name = config.run_name or _auto_run_name(config)
    root = Path(config.output_root) / run_name
    paths = {
        "run": root,
        "extractions": root / "extractions",
        "evaluation": root / "evaluation",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Save run config for reproducibility
    (root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return paths


def _write_excel_autosize(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to Excel with simple auto-widths; falls back silently if openpyxl missing.
    """
    try:
        from openpyxl.utils import get_column_letter
    except ModuleNotFoundError as exc:
        print(f"Warning: {exc}. Skipping Excel export; CSV was still written.")
        return

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet_name = "Sheet1"
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns, 1):
            values = df[col].fillna("").astype(str).tolist()
            max_len = max([len(str(col))] + [len(v) for v in values])
            ws.column_dimensions[get_column_letter(idx)].width = min(max_len + 2, 60)


def _find_pdf_path(pmcid: int, pdf_folder: str) -> Optional[Path]:
    primary = Path(pdf_folder) / f"{pmcid}.pdf"
    if primary.exists():
        return primary
    alt = Path(pdf_folder) / f"PMCID{pmcid}.pdf"
    if alt.exists():
        return alt
    return None


def _is_retryable_error(exc: Exception) -> bool:
    """
    Identify transient LLM errors where a retry makes sense (e.g., overloaded/unavailable).
    """
    text = str(exc).lower()
    retry_terms = (
        "overloaded",
        "unavailable",
        "try again later",
        "temporarily unavailable",
        "resource exhausted",
        "rate limit",
    )
    if any(term in text for term in retry_terms):
        return True

    code = getattr(exc, "code", None)
    try:
        code = code() if callable(code) else code
    except Exception:
        code = None
    if code and any(token in str(code).lower() for token in ("unavailable", "resource_exhausted")):
        return True

    return False


def _generate_with_retry(
    client: LLMClient,
    prompt: str,
    pdf_paths: List[str],
    max_tokens: Optional[int],
    label: str,
    max_attempts: int = 3,
    backoff_seconds: int = 5,
) -> str:
    """
    Call the LLM with a small retry loop for transient overload errors.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        attempt_label = label if attempt == 1 else f"{label} (retry {attempt}/{max_attempts})"
        try:
            with Spinner(attempt_label):
                return client.generate(prompt, pdf_paths=pdf_paths, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_error(exc):
                break
            wait_time = backoff_seconds * attempt
            print(f"Transient LLM error ({exc}); retrying in {wait_time}s...")
            time.sleep(wait_time)
    if last_exc:
        raise last_exc
    raise RuntimeError("LLM call failed without raising an exception")


def run_experiments(config: ExperimentConfig) -> Dict:
    folders = _prepare_output_folders(config)
    annotations = load_annotations(config.gold_path)
    pmcids = config.pmcids or list_pmcids(config.pdf_folder)
    if not pmcids:
        raise SystemExit(f"No PDFs found in {config.pdf_folder}")

    client = LLMClient(model=config.model, temperature=config.temperature)

    stats = {"total": len(pmcids), "success": 0, "failed": 0, "skipped": 0}
    failures: List[tuple[int, str]] = []

    for idx, pmcid in enumerate(pmcids, start=1):
        ico_rows = get_icos(pmcid, annotations=annotations)
        pdf_paths: List[str] = []
        if config.fewshot_pdf_paths:
            pdf_paths.extend([str(Path(p)) for p in config.fewshot_pdf_paths])
        pdf_path_main = _find_pdf_path(pmcid, config.pdf_folder)
        if pdf_path_main:
            pdf_paths.append(str(pdf_path_main))
        else:
            # No main PDF; if no few-shot PDFs either, fail
            if not pdf_paths:
                failures.append((pmcid, "PDF not found"))
                stats["failed"] += 1
                continue

        article_text = None

        prompt_template_path = config.prompt_template
        if config.fewshot_pdf_paths and config.fewshot_prompt_template:
            prompt_template_path = config.fewshot_prompt_template

        prompt = build_prompt(article_text=article_text, ico_rows=ico_rows, base_prompt_path=prompt_template_path)

        raw_response = ""
        try:
            label = f"[{idx}/{len(pmcids)}] PMCID={pmcid} extracting…"
            raw_response = _generate_with_retry(
                client=client,
                prompt=prompt,
                pdf_paths=pdf_paths,
                max_tokens=config.max_tokens,
                label=label,
            )
            result = ExtractionResult.from_response(
                pmcid=pmcid,
                response=raw_response,
                model=config.model,
                prompt_strategy=config.prompt_label or Path(config.prompt_template).stem,
            )
            suffix = config.prompt_label or Path(config.prompt_template).stem
            out_name = f"{pmcid}_{suffix}.jsonl"
            result.save(folders["extractions"] / out_name)
            stats["success"] += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            failures.append((pmcid, str(exc)))
            log_path = folders["logs"] / f"{pmcid}_error.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"Prompt:\\n{prompt}\\n\\nResponse:\\n{raw_response}\\n\\nError: {exc}", encoding="utf-8")

    # Evaluation (always, per flowchart)
    eval_results = None
    if stats["success"] > 0:
        evaluator = BatchEvaluator(
            gold_path=config.gold_path,
            output_dir=str(folders["evaluation"]),
        )
        eval_results = evaluator.evaluate_directory(
            predictions_dir=str(folders["extractions"]),
            suffix_filter=f"_{config.prompt_label or Path(config.prompt_template).stem}",
            run_name=config.run_name or folders["run"].name,
        )

    # Export Excel/CSV
    rows = load_extractions(folders["extractions"], suffix=f"_{config.prompt_label or Path(config.prompt_template).stem}")
    if rows:
        df = pd.DataFrame(rows, columns=COLUMNS)
        excel_path = folders["run"] / "results.xlsx"
        csv_path = folders["run"] / "results.csv"
        _write_excel_autosize(df, excel_path)
        df.to_csv(csv_path, index=False)

    summary = {
        "stats": stats,
        "failures": failures,
        "eval": eval_results,
        "run_folder": str(folders["run"]),
    }
    (folders["run"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RCT extraction experiments without LangExtract.")
    parser.add_argument("--model", default=None, help="Model name (e.g., gpt-5.1-mini)")
    parser.add_argument("--prompt-label", default=None, help="Short name for the prompt/template (used in filenames)")
    parser.add_argument("--pdf-folder", default=None, help="Folder with PDFs")
    parser.add_argument("--gold-path", default=None, help="Path to gold annotations JSON")
    parser.add_argument("--prompt-template", default=None, help="Prompt template path")
    parser.add_argument("--output-root", default=None, help="Root folder for outputs")
    parser.add_argument("--run-name", default=None, help="Optional run name (folder will be created)")
    parser.add_argument("--pmcids", nargs="+", type=int, default=None, help="Optional list of PMCIDs to process")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for the model response")
    parser.add_argument("--fewshot-pdf-paths", nargs="+", default=None, help="Optional list of PDF paths for few-shot context")
    parser.add_argument(
        "--fewshot-prompt-template",
        default=None,
        help="Optional prompt template path to use when few-shot PDFs are provided",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = ExperimentConfig.from_args(args)
    summary = run_experiments(config)

    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Run folder: {summary['run_folder']}")
    print(f"Processed: {summary['stats']['success']}/{summary['stats']['total']}")
    if summary["failures"]:
        print("Failures:")
        for pmcid, reason in summary["failures"]:
            print(f"  - {pmcid}: {reason}")
    if summary["eval"]:
        print("Evaluation (micro F1):", summary["eval"].get("micro_f1"))


if __name__ == "__main__":
    main()
