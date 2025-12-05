import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import pandas as pd

from utils import get_pmcid_from_filename


# Map LangExtract classes to spreadsheet columns
CLASS_TO_COLUMN = {
    "intervention_group_size": "intervention_group_size",
    "comparator_group_size": "comparator_group_size",
    "intervention_events": "intervention_events",
    "comparator_events": "comparator_events",
    "intervention_rate": "intervention_rate",
    "comparator_rate": "comparator_rate",
    "intervention_mean": "intervention_mean",
    "comparator_mean": "comparator_mean",
    "intervention_standard_deviation": "intervention_standard_deviation",
    "comparator_standard_deviation": "comparator_standard_deviation",
}

# Column order for the Excel sheet
COLUMNS = [
    "pmcid",
    "outcome",
    "intervention",
    "comparator",
    "outcome_type",
    "intervention_events",
    "intervention_group_size",
    "comparator_events",
    "comparator_group_size",
    "intervention_mean",
    "intervention_standard_deviation",
    "comparator_mean",
    "comparator_standard_deviation",
    "intervention_rate",
    "comparator_rate",
    "source_file",
]


def _first_non_null(attrs: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """Return the first non-null attribute value for any of the candidate keys."""
    for key in keys:
        if key in attrs and attrs[key] is not None:
            return attrs[key]
    return None


def _merge_cell(current: Any, new_val: Any) -> Any:
    """Prefer the first value; keep existing if already set."""
    if new_val is None:
        return current
    new_val = str(new_val).strip()
    if not new_val:
        return current
    if current is None or str(current).strip() == "":
        return new_val
    return current


def load_extractions(extractions_dir: Path, suffix: str | None = None) -> List[Dict[str, Any]]:
    """
    Load JSONL outputs (new direct schema or legacy LangExtract) and aggregate them into rows keyed by (pmcid, intervention, comparator, outcome).

    Args:
        extractions_dir: Directory that contains *.jsonl files (one per PMCID).
        suffix: Optional filename suffix filter (e.g., "_guided" or "_all"). If provided, only files ending with
                `{suffix}.jsonl` are loaded.
    """
    rows: List[Dict[str, Any]] = []

    for jsonl_path in sorted(extractions_dir.glob("*.jsonl")):
        if suffix and not jsonl_path.name.endswith(f"{suffix}.jsonl"):
            continue

        pmcid = get_pmcid_from_filename(jsonl_path.name)
        triplets: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)

                ex_list = doc.get("extractions", [])

                # Case 0: new direct-row schema (one dict per ICO row, no extraction_class)
                if ex_list and all("extraction_class" not in ex for ex in ex_list if isinstance(ex, dict)):
                    for entry in ex_list:
                        row = {
                            "pmcid": pmcid,
                            "outcome": entry.get("outcome"),
                            "intervention": entry.get("intervention"),
                            "comparator": entry.get("comparator"),
                            "outcome_type": entry.get("outcome_type"),
                            "intervention_group_size": entry.get("intervention_group_size"),
                            "comparator_group_size": entry.get("comparator_group_size"),
                            "intervention_events": entry.get("intervention_events"),
                            "comparator_events": entry.get("comparator_events"),
                            "intervention_rate": entry.get("intervention_rate"),
                            "comparator_rate": entry.get("comparator_rate"),
                            "intervention_mean": entry.get("intervention_mean"),
                            "comparator_mean": entry.get("comparator_mean"),
                            "intervention_standard_deviation": entry.get("intervention_standard_deviation"),
                            "comparator_standard_deviation": entry.get("comparator_standard_deviation"),
                            "source_file": jsonl_path.name,
                        }
                        rows.append(row)
                    continue

                # Case 1: new full-row schema (`rct_row`)
                for extraction in ex_list:
                    ex_class = extraction.get("extraction_class")
                    if ex_class == "rct_row":
                        payload = extraction.get("attributes") or {}
                        text = extraction.get("extraction_text")
                        if text:
                            try:
                                payload = json.loads(text)
                            except json.JSONDecodeError:
                                pass

                        row = {
                            "pmcid": payload.get("pmcid", pmcid),
                            "outcome": payload.get("outcome"),
                            "intervention": payload.get("intervention"),
                            "comparator": payload.get("comparator"),
                            "outcome_type": payload.get("outcome_type"),
                            "intervention_group_size": payload.get("intervention_group_size"),
                            "comparator_group_size": payload.get("comparator_group_size"),
                            "intervention_events": payload.get("intervention_events"),
                            "comparator_events": payload.get("comparator_events"),
                            "intervention_rate": payload.get("intervention_rate"),
                            "comparator_rate": payload.get("comparator_rate"),
                            "intervention_mean": payload.get("intervention_mean"),
                            "comparator_mean": payload.get("comparator_mean"),
                            "intervention_standard_deviation": payload.get("intervention_standard_deviation"),
                            "comparator_standard_deviation": payload.get("comparator_standard_deviation"),
                            "source_file": jsonl_path.name,
                        }
                        rows.append(row)
                if any(extraction.get("extraction_class") == "rct_row" for extraction in ex_list):
                    continue  # already handled this line

                # Case 2: legacy per-field schema with group_index (group == one ICO row)
                if any(extraction.get("group_index") is not None for extraction in ex_list):
                    grouped: Dict[int, Dict[str, Any]] = {}
                    for extraction in ex_list:
                        gi = extraction.get("group_index")
                        if gi is None:
                            continue
                        ex_class = extraction.get("extraction_class")
                        val = extraction.get("extraction_text")
                        if isinstance(val, str) and val.strip().lower() == "none":
                            val = None

                        attrs = extraction.get("attributes") or {}
                        outcome_attr = _first_non_null(attrs, ("Outcome", "outcome"))
                        intervention_attr = _first_non_null(attrs, ("Intervention", "intervention"))
                        comparator_attr = _first_non_null(attrs, ("Comparator", "comparator"))
                        # If timepoint exists, append to outcome to disambiguate
                        timepoint = attrs.get("Timepoint") or attrs.get("timepoint")
                        if outcome_attr and timepoint:
                            outcome_attr = f"{outcome_attr} ({timepoint})"

                            row = grouped.setdefault(
                                gi,
                                {
                                    "pmcid": pmcid,
                                    "outcome": None,
                                    "intervention": None,
                                    "comparator": None,
                                    "outcome_type": None,
                                    "intervention_group_size": None,
                                    "comparator_group_size": None,
                                    "intervention_events": None,
                                "comparator_events": None,
                                "intervention_rate": None,
                                "comparator_rate": None,
                                "intervention_mean": None,
                                "comparator_mean": None,
                                "intervention_standard_deviation": None,
                                "comparator_standard_deviation": None,
                                "source_file": jsonl_path.name,
                            },
                        )

                        # Populate identity fields from attributes when present
                        if outcome_attr and not row["outcome"]:
                            row["outcome"] = outcome_attr
                        if intervention_attr and not row["intervention"]:
                            row["intervention"] = intervention_attr
                        if comparator_attr and not row["comparator"]:
                            row["comparator"] = comparator_attr

                        if ex_class == "outcome":
                            row["outcome"] = val
                        elif ex_class == "intervention":
                            row["intervention"] = val
                        elif ex_class == "comparator":
                            row["comparator"] = val
                        elif ex_class in CLASS_TO_COLUMN:
                            row[CLASS_TO_COLUMN[ex_class]] = val

                    # keep rows that have at least one data field
                    for row in grouped.values():
                        has_data = any(
                            row.get(col) is not None
                            for col in CLASS_TO_COLUMN.values()
                        )
                        if has_data:
                            rows.append(row)
                    continue

                # Case 3: legacy per-field schema (attributes-based grouping)
                for extraction in ex_list:
                    ex_class = extraction.get("extraction_class")
                    col = CLASS_TO_COLUMN.get(ex_class)
                    if not col:
                        continue  # ignore extraction classes we are not exporting

                    attrs = extraction.get("attributes") or {}
                    outcome = _first_non_null(attrs, ("Outcome", "outcome"))
                    intervention = _first_non_null(attrs, ("Intervention", "intervention"))
                    comparator = _first_non_null(attrs, ("Comparator", "comparator"))

                    # Each unique (I, C, O) becomes its own row
                    key = (intervention, comparator, outcome)
                    if key not in triplets:
                        triplets[key] = {
                            "pmcid": pmcid,
                            "intervention": intervention,
                            "comparator": comparator,
                            "outcome": outcome,
                            "source_file": jsonl_path.name,
                        }

                    triplets[key][col] = _merge_cell(triplets[key].get(col), extraction.get("extraction_text"))

        rows.extend(triplets.values())

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Export extraction JSONL outputs to an Excel file (one row per ICO triplet)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the extractions directory (e.g., outputs/my_run/extractions)",
    )
    parser.add_argument(
        "--output",
        default="extractions.xlsx",
        help="Destination .xlsx file. Defaults to ./extractions.xlsx",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional filename suffix to filter JSONL files (e.g., '_guided' or '_all'). Do not include '.jsonl'.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    rows = load_extractions(input_dir, suffix=args.suffix)
    if not rows:
        raise SystemExit("No rows to export. Check the input path and suffix filter.")

    df = pd.DataFrame(rows, columns=COLUMNS)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
