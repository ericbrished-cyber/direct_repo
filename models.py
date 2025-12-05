from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

NUMERIC_FIELDS = [
    "intervention_group_size",
    "comparator_group_size",
    "intervention_events",
    "comparator_events",
    "intervention_rate",
    "comparator_rate",
    "intervention_mean",
    "comparator_mean",
    "intervention_standard_deviation",
    "comparator_standard_deviation",
]

IDENTITY_FIELDS = ["outcome", "intervention", "comparator"]

# Map per-field extraction classes to direct-row fields
EXTRACTION_CLASS_TO_FIELD = {
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


def _coerce_numeric(value: Any) -> Any:
    """Best-effort numeric coercion; leave untouched on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    # Strip common trailing symbols (%, commas)
    cleaned = text.replace(",", "")
    cleaned = cleaned[:-1] if cleaned.endswith("%") else cleaned
    try:
        return float(cleaned)
    except ValueError:
        return text


def _extract_json_snippet(text: str) -> str:
    """Extract the first JSON object from a text blob."""
    if "```" in text:
        parts = re.split(r"```(?:json)?", text, flags=re.IGNORECASE)
        if len(parts) >= 3:
            text = parts[1]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


@dataclass
class OutcomeExtraction:
    outcome: str
    intervention: str
    comparator: str
    intervention_group_size: Optional[Any] = None
    comparator_group_size: Optional[Any] = None
    intervention_events: Optional[Any] = None
    comparator_events: Optional[Any] = None
    intervention_rate: Optional[Any] = None
    comparator_rate: Optional[Any] = None
    intervention_mean: Optional[Any] = None
    comparator_mean: Optional[Any] = None
    intervention_standard_deviation: Optional[Any] = None
    comparator_standard_deviation: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OutcomeExtraction":
        data = {k: payload.get(k) for k in IDENTITY_FIELDS + NUMERIC_FIELDS}
        missing = [k for k in IDENTITY_FIELDS if not data.get(k)]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        for key in NUMERIC_FIELDS:
            data[key] = _coerce_numeric(data[key])
        extra = {k: v for k, v in payload.items() if k not in data}
        return cls(extra=extra, **data)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(payload.pop("extra", {}) or {})
        return {k: v for k, v in payload.items() if v is not None}


@dataclass
class ExtractionResult:
    pmcid: int
    prompt_strategy: str
    model: str
    extractions: List[OutcomeExtraction] = field(default_factory=list)
    raw_response: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_response(
        cls,
        pmcid: int,
        response: str | Dict[str, Any],
        model: str,
        prompt_strategy: str,
    ) -> "ExtractionResult":
        def _attrs_get(attrs: Dict[str, Any], key: str) -> Any:
            return attrs.get(key) if key in attrs else attrs.get(key.lower())

        def _convert_field_schema(items: List[Dict[str, Any]]) -> tuple[List["OutcomeExtraction"], List[str]]:
            grouped: Dict[Any, Dict[str, Any]] = {}
            errs: List[str] = []

            for item in items:
                if not isinstance(item, dict):
                    errs.append(f"Ignoring non-dict extraction {item!r}")
                    continue

                ex_class = item.get("extraction_class")
                field_name = EXTRACTION_CLASS_TO_FIELD.get(ex_class)
                if not field_name:
                    errs.append(f"Unrecognized extraction_class {ex_class!r}")
                    continue

                attrs = item.get("attributes") or {}
                outcome = _attrs_get(attrs, "Outcome")
                intervention = _attrs_get(attrs, "Intervention")
                comparator = _attrs_get(attrs, "Comparator")
                extra_attrs = {
                    k: v
                    for k, v in attrs.items()
                    if k not in {"Outcome", "outcome", "Intervention", "intervention", "Comparator", "comparator"}
                }

                key = (
                    outcome,
                    intervention,
                    comparator,
                    tuple(sorted((k, str(v)) for k, v in extra_attrs.items())),
                )
                row = grouped.setdefault(
                    key,
                    {
                        "outcome": outcome,
                        "intervention": intervention,
                        "comparator": comparator,
                        "extra": dict(extra_attrs),
                    },
                )
                # Keep any new extra attributes that were not seen before
                for k, v in extra_attrs.items():
                    row["extra"].setdefault(k, v)

                raw_val = item.get("value", item.get("extraction_text"))
                row[field_name] = _coerce_numeric(raw_val)

            parsed_rows: List[OutcomeExtraction] = []
            for row in grouped.values():
                payload = {k: v for k, v in row.items() if k != "extra"}
                payload.update(row.get("extra") or {})
                try:
                    parsed_rows.append(OutcomeExtraction.from_dict(payload))
                except Exception as exc:  # noqa: BLE001
                    errs.append(f"Could not parse grouped extraction {payload}: {exc}")

            return parsed_rows, errs

        if isinstance(response, dict):
            data = response
            raw_text = json.dumps(response)
        else:
            raw_text = response
            snippet = _extract_json_snippet(response)
            data = json.loads(snippet)

        items = data.get("extractions", [])
        if not isinstance(items, list):
            raise ValueError("Response must contain an 'extractions' list")

        parsed: List[OutcomeExtraction] = []
        errors: List[str] = []

        # Newer prompts may emit per-field schema with `extraction_class` and `attributes`
        if items and all(isinstance(item, dict) and "extraction_class" in item for item in items):
            parsed, errors = _convert_field_schema(items)
        else:
            for item in items:
                try:
                    parsed.append(OutcomeExtraction.from_dict(item))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Could not parse extraction {item}: {exc}")

        return cls(
            pmcid=int(pmcid),
            prompt_strategy=prompt_strategy,
            model=model,
            extractions=parsed,
            raw_response=raw_text,
            errors=errors,
        )

    def to_prediction_dict(self) -> Dict[str, Any]:
        return {
            "pmcid": self.pmcid,
            "prompt_strategy": self.prompt_strategy,
            "model": self.model,
            "extractions": [ex.to_dict() for ex in self.extractions],
            "errors": self.errors,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_prediction_dict(), ensure_ascii=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl() + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ExtractionResult":
        text = Path(path).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"No content in {path}")
        first_line = text.splitlines()[0]
        data = json.loads(first_line)
        items = [OutcomeExtraction.from_dict(item) for item in data.get("extractions", [])]
        return cls(
            pmcid=data.get("pmcid"),
            prompt_strategy=data.get("prompt_strategy", "unknown"),
            model=data.get("model", "unknown"),
            extractions=items,
            raw_response=data.get("raw_response"),
            errors=data.get("errors", []),
        )
