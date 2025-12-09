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
    "intervention_mean",
    "comparator_mean",
    "intervention_standard_deviation",
    "comparator_standard_deviation",
]

IDENTITY_FIELDS = ["outcome", "intervention", "comparator"]

@dataclass
class OutcomeExtraction:
    outcome: str
    intervention: str
    comparator: str
    intervention_group_size: Optional[float] = None
    comparator_group_size: Optional[float] = None
    intervention_events: Optional[float] = None
    comparator_events: Optional[float] = None
    intervention_mean: Optional[float] = None
    comparator_mean: Optional[float] = None
    intervention_standard_deviation: Optional[float] = None
    comparator_standard_deviation: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OutcomeExtraction":
        # Extract identity fields
        data = {}
        for k in IDENTITY_FIELDS:
            if k not in payload:
                 raise ValueError(f"Missing required field: {k}")
            data[k] = payload[k]

        # Extract numeric fields
        for k in NUMERIC_FIELDS:
            val = payload.get(k)
            if val is not None:
                try:
                    # Simple numeric conversion
                    data[k] = float(val)
                except (ValueError, TypeError):
                    data[k] = None
            else:
                data[k] = None

        # Store anything else in extra
        extra = {k: v for k, v in payload.items() if k not in data}
        return cls(extra=extra, **data)

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
        # 1. Parse JSON
        if isinstance(response, dict):
            data = response
            raw_text = json.dumps(response)
        else:
            raw_text = response
            # Simple JSON extraction regex
            match = re.search(r"(\{.*\})", response.replace("\n", " "), re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    # Fallback: try to find the largest bracketed block
                    # But if the regex failed to find valid JSON, we likely can't do much.
                    # Let's try the whole string just in case.
                    try:
                        data = json.loads(response)
                    except json.JSONDecodeError:
                        return cls(pmcid=pmcid, prompt_strategy=prompt_strategy, model=model, raw_response=raw_text, errors=["Could not parse JSON"])
            else:
                 # Try parsing the whole string directly
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                     return cls(pmcid=pmcid, prompt_strategy=prompt_strategy, model=model, raw_response=raw_text, errors=["No JSON object found"])

        # 2. Extract items
        items = data.get("extractions", [])
        if not isinstance(items, list):
            return cls(pmcid=pmcid, prompt_strategy=prompt_strategy, model=model, raw_response=raw_text, errors=["Response missing 'extractions' list"])

        # 3. Convert to objects
        parsed: List[OutcomeExtraction] = []
        errors: List[str] = []

        for item in items:
            try:
                parsed.append(OutcomeExtraction.from_dict(item))
            except Exception as exc:
                errors.append(f"Invalid item {item}: {exc}")

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
