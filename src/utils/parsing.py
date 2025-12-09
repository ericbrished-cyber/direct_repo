import json
import re
from typing import Any, Dict, List, Union

def clean_and_parse_json(raw_text: str) -> Union[Dict, List, None]:
    """
    Robustly find JSON content using regex (e.g., between ```json blocks or first [ / last ]).
    Handle malformed JSON gracefully.
    """
    if not raw_text:
        return None

    # 1. Try extracting Markdown code blocks first
    json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_block_pattern, raw_text)

    candidate_text = raw_text
    if match:
        candidate_text = match.group(1)

    # 2. If parsing fails or no block, try finding the outermost bracket pair
    # We look for either { ... } or [ ... ]
    # This simple regex looks for the first [ or { and the last ] or }
    try:
        cleaned_text = candidate_text.strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Fallback: Extract from first [ to last ] or first { to last }
    # We prioritize list [ ] if we expect a list of ICOs

    # Find start and end indices
    first_brace = candidate_text.find('{')
    last_brace = candidate_text.rfind('}')
    first_bracket = candidate_text.find('[')
    last_bracket = candidate_text.rfind(']')

    # Determine which pair is outer-most or seemingly correct
    # If we expect a list, we prioritize brackets.
    start = -1
    end = -1

    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        start = first_bracket
        end = last_bracket + 1
    elif first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        start = first_brace
        end = last_brace + 1

    if start != -1:
        extracted = candidate_text[start:end]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            # Try some basic cleanup (e.g., trailing commas) if needed,
            # but for now let's return None or handle specific cases if they appear.
            pass

    # As a last resort, try to be very aggressive?
    # For now, return None or an empty list to indicate failure.
    print(f"Failed to parse JSON from text: {raw_text[:100]}...")
    return None
