from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _as_template_string(path: str | Path) -> str:
    raw = Path(path).read_text(encoding="utf-8")
    # Allow YAML-formatted prompt files with a "prompt_description" key.
    try:
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict) and "prompt_description" in parsed:
            return str(parsed["prompt_description"])
    except Exception:
        pass
    return raw


def format_ico_schema(ico_rows: Iterable[Dict[str, Any]]) -> str:
    """
    Turn a list of ICO rows into a compact bullet list for prompting.
    """
    lines: List[str] = []
    for row in ico_rows:
        outcome = row.get("outcome") or "<outcome>"
        intervention = row.get("intervention") or "<intervention>"
        comparator = row.get("comparator") or "<comparator>"
        outcome_type = row.get("outcome_type")
        line = f"- {outcome} (Intervention: {intervention} vs Comparator: {comparator})"
        if outcome_type:
            line += f" — {outcome_type}"
        lines.append(line)
    return "\n".join(lines) if lines else "- No ICO schema available for this PMCID."


def build_prompt(
    article_text: Optional[str],
    ico_rows: Iterable[Dict[str, Any]],
    base_prompt_path: str,
    max_article_chars: int = 18_000,
) -> str:
    """
    Assemble a full prompt from template + ICO schema + optional article text.
    """
    base = _as_template_string(base_prompt_path).strip()
    schema_block = format_ico_schema(ico_rows)

    parts = [base, "ICO schema for this article:\n" + schema_block]
    if article_text:
        article = article_text.strip()
        if len(article) > max_article_chars:
            article = article[:max_article_chars] + "\n\n[truncated]"
        parts.append("Article text:\n" + article)
    else:
        parts.append("The full article PDF is attached as context. Use it as the primary source.")

    return "\n\n".join(parts)
