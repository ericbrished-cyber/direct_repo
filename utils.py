import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

PMCID_RE = re.compile(r"(?:PMCID)?(\d{6,8})", re.IGNORECASE)
DEFAULT_GOLD_PATH = Path("gold-standard/annotated_rct_dataset.json")


def list_pmcids(pdf_folder: str) -> List[int]:
    """
    Return all PMCIDs (as ints) present in a PDF folder.
    Matches both `1234567.pdf` and `PMCID1234567.pdf`.
    """
    pmcids: list[int] = []
    seen: set[int] = set()
    for pdf_path in Path(pdf_folder).glob("*.pdf"):
        match = PMCID_RE.search(pdf_path.stem)
        if not match:
            continue
        pmcid = int(match.group(1))
        if pmcid not in seen:
            seen.add(pmcid)
            pmcids.append(pmcid)
    return sorted(pmcids)


@lru_cache(maxsize=1)
def load_annotations(gold_path: Path | str = DEFAULT_GOLD_PATH) -> List[Dict[str, Any]]:
    """Load the gold-standard annotations once and cache them."""
    path = Path(gold_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold-standard file not found: {path}")
    return json.loads(path.read_text())


def get_icos(
    pmcid: int | str,
    annotations: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return all ICO rows for a given PMCID from the gold annotations."""
    rows = annotations if annotations is not None else load_annotations()
    pmcid_int = int(pmcid)
    return [row for row in rows if int(row.get("pmcid", -1)) == pmcid_int]


def get_prompt_all(prompt_path: str = "prompt_templates/all_prompt_new.md") -> str:
    """Load the base (non-guided) prompt from disk."""
    path = Path(prompt_path)
    return path.read_text(encoding="utf-8")


def get_fulltext(pmcid: int | str, text_folder_path: str = "data/Markdown") -> str:
    """
    Load the markdown content for a PMCID. Returns a friendly message if missing.
    """
    md_file_path = Path(text_folder_path) / f"{pmcid}.md"
    if md_file_path.exists():
        return md_file_path.read_text(encoding="utf-8")
    return f"Markdown file for PMCID {pmcid} not found in {text_folder_path}."


def ensure_markdown(
    pmcid: int | str,
    pdf_folder: str,
    markdown_folder: str = "data/Markdown",
) -> Optional[str]:
    """
    Ensure a markdown file exists for the pmcid by converting the PDF if needed.
    Returns the markdown path or None if conversion failed.
    """
    md_file_path = Path(markdown_folder) / f"{pmcid}.md"
    if md_file_path.exists():
        return str(md_file_path)

    pdf_path = Path(pdf_folder) / f"{pmcid}.pdf"
    if not pdf_path.exists():
        alt = Path(pdf_folder) / f"PMCID{pmcid}.pdf"
        pdf_path = alt if alt.exists() else None
    if pdf_path is None or not pdf_path.exists():
        return None

    from pdf_converter import convert_pdf_to_markdown

    return convert_pdf_to_markdown(pdf_path, output_dir=markdown_folder)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Convenience loader for small YAML config/example files."""
    return yaml.safe_load(Path(path).read_text()) or {}


def simplified_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the ICO identity fields from a gold row."""
    return {
        "id": entry["id"],
        "pmcid": entry["pmcid"],
        "intervention": entry["intervention"],
        "comparator": entry["comparator"],
        "outcome": entry["outcome"],
        "outcome_type": entry.get("outcome_type"),
    }


def get_pmcid_from_filename(path: str) -> int:
    """
    Extract PMCID as the leading digit sequence in the filename (ignores digits in suffixes).
    E.g. '4132222_guided.jsonl' -> 4132222, 'PMCID4132222_gpt5direct.jsonl' -> 4132222
    """
    name = Path(path).name
    match = re.match(r"^(?:PMCID)?(\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"No leading pmcid digits in filename: {name}")
    return int(match.group(1))


def get_events_from_rate(event_rate: Any, n: Any) -> Optional[int]:
    """
    Estimate events from an event rate and sample size.
    Accepts proportions (0-1) or percentages (0-100).
    """
    if event_rate is None or n is None:
        return None
    try:
        rate = float(event_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid event_rate: {event_rate}") from exc
    if rate > 1:
        rate = rate / 100.0
    return int(round(rate * int(n)))
