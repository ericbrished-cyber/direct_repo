import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PMCID_RE = re.compile(r"(?:PMCID)?(\d{5,8})", re.IGNORECASE)
DEFAULT_GOLD_PATH = Path("gold-standard/gold_standard_clean.json")


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
def load_annotations(gold_path: Union[Path, str] = DEFAULT_GOLD_PATH) -> List[Dict[str, Any]]:
    """Load the gold-standard annotations once and cache them."""
    path = Path(gold_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold-standard file not found: {path}")
    return json.loads(path.read_text())


def get_icos(
    pmcid: Union[int, str],
    annotations: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return ICO rows (intervention, comparator, outcome only) for a given PMCID."""
    rows = annotations if annotations is not None else load_annotations()
    pmcid_int = int(pmcid)
    return [
        {
            "intervention": row.get("intervention"),
            "comparator": row.get("comparator"),
            "outcome": row.get("outcome"),
        }
        for row in rows
        if int(row.get("pmcid", -1)) == pmcid_int
    ]


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
