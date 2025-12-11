import json
from pathlib import Path
from typing import List, Dict, Any


def dump_debug_json(name: str, payload: Any) -> Path:
    """Write JSON debug payload to data/debug/{name}.json."""
    debug_path = Path("data/debug") / f"{name}.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[DRY-RUN] Wrote {debug_path}")
    return debug_path


def clean_claude_messages(messages: List[Dict]) -> List[Dict]:
    """Strip base64 blobs from Claude message payload."""
    cleaned = []
    for msg in messages:
        msg_copy = {"role": msg.get("role"), "content": []}
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "document":
                src = block.get("source", {})
                block_copy = {
                    "type": "document",
                    "source": {
                        "type": src.get("type"),
                        "media_type": src.get("media_type"),
                        "data": "<omitted>",
                    },
                }
                if block.get("cache_control"):
                    block_copy["cache_control"] = block["cache_control"]
                msg_copy["content"].append(block_copy)
            else:
                msg_copy["content"].append(block)
        cleaned.append(msg_copy)
    return cleaned


def clean_gpt_messages(messages: List[Dict]) -> List[Dict]:
    """Strip base64 blobs from GPT message payload."""
    cleaned = []
    for msg in messages:
        msg_copy = {"role": msg.get("role"), "content": []}
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "input_file":
                block_copy = block.copy()
                block_copy["file_data"] = "<omitted>"
                msg_copy["content"].append(block_copy)
            else:
                msg_copy["content"].append(block)
        cleaned.append(msg_copy)
    return cleaned
