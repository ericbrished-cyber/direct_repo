You are extracting pairwise ICO results from a randomized controlled trial.
Return only JSON of the form: {"extractions":[ ... ]}.

Each item in "extractions" is one ICO row with exactly these fields (use null for missing/not applicable):

{
  "id": null,
  "evidence_inference_prompt_id": null,
  "pmcid": "<STRING OR INTEGER>",
  "outcome": "<STRING>",
  "intervention": "<STRING>",
  "comparator": "<STRING>",
  "outcome_type": "<continuous | binary>",
  "intervention_events": null,
  "intervention_group_size": null,
  "comparator_events": null,
  "comparator_group_size": null,
  "intervention_mean": null,
  "intervention_standard_deviation": null,
  "comparator_mean": null,
  "comparator_standard_deviation": null
}

---------------------------------------------------------------------
FIXED ICO TRIPLETS (MUST NOT BE ALTERED)

The ICO triplets provided in {ico_list} are fixed and authoritative. You must:

- use them exactly as written,
- never modify their wording,
- never expand, merge, split, or normalize them,
- never create ANY new ICO triplets,
- never change intervention, comparator, or outcome names,
- never infer additional outcomes or arms from the text.

If the study reports similar or related outcomes, ignore them unless they match an ICO triplet exactly.

---------------------------------------------------------------------
STRICT ITERATION LOGIC (ABSOLUTELY REQUIRED)

Treat {ico_list} as the *complete and final* set of ICO triplets.

Your task:

1. **Iterate through the ICO triplets in {ico_list}, in the exact order they appear.**
2. For each ICO triplet:
   - Search the article ONLY for numeric information corresponding to that exact triplet.
   - If at least one numeric value exists → produce exactly one JSON object for that triplet.
   - If no numeric values exist → produce no JSON object for that triplet.
3. **You must NOT generate any ICO extraction for outcomes or arms not in {ico_list}.**
4. Your final JSON must contain **AT MOST len({ico_list}) objects**.

You are NOT allowed to scan the article for all outcomes.
You may only scan the article for evidence that matches each triplet in {ico_list} verbatim.

---------------------------------------------------------------------
EXTRACTION RULES

Continuous outcomes: extract when explicitly reported
- intervention_group_size
- comparator_group_size
- intervention_mean
- comparator_mean
- intervention_standard_deviation
- comparator_standard_deviation

Binary outcomes: extract when explicitly reported
- intervention_group_size
- comparator_group_size
- intervention_events
- comparator_events
- numeric rates (if reported)

---------------------------------------------------------------------
ALLOWED INFERENCE (EVENTS FROM RATE × N)

If only rate (%) and group size are reported, you may infer:

events = round((rate / 100) * group_size)

Ensure 0 ≤ events ≤ group_size.
Keep the original numeric rate in a *_rate field if present.

Do NOT infer means, SDs, or group sizes.

---------------------------------------------------------------------
GENERAL RULES

- Use plain numbers only (no %, no units).
- Use only information from Abstract and Results.
- If a numeric value is missing, set it to null.
- If a numeric value is ambiguous and cannot be confidently extracted, set it to "?".
- Output raw JSON only (no markdown, no text, no comments).

---------------------------------------------------------------------
ICO UNIQUENESS (CRITICAL)

You must produce at most one JSON extraction per ICO triplet.

If multiple mentions of the same value exist → extract it once.
If slightly different values exist → choose the most precise or the value from the main Results section.

---------------------------------------------------------------------
TIMEPOINT RULES

- If the ICO triplet specifies a timepoint → use only that timepoint.
- If not specified:
  - prefer post-intervention values,
  - if multiple post-intervention values exist → choose the latest one.

---------------------------------------------------------------------
MISSING DATA

- If no numeric values exist for an ICO triplet → omit that triplet entirely.
- If at least one numeric field exists but others are unclear → include the triplet and set unclear fields to "?".

---------------------------------------------------------------------
FINAL RULE

Do NOT output any ICO triplets other than those explicitly listed in {ico_list}.
