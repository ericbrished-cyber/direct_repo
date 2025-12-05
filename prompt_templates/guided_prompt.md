# Prompt
You are extracting **pairwise ICO results** from a randomized controlled trial. Return **only** JSON of the form: `{"extractions":[ ... ]}`.

Each item in "extractions" is one ICO row with EXACTLY these fields (use null for missing/not applicable):
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

## What to extract
You are given a list of target ICO triplets:

{ico_list}

Each triplet is `(Intervention, Comparator, Outcome)` (optionally with extra disambiguators like Timepoint/Population).

For **each such ICO triplet, and only these**:
- If the outcome is continuous, fill the mean/SD/group sizes when explicitly reported; leave other fields null.
- If the outcome is binary, fill events/group sizes (and rates if explicitly reported); leave other fields null.

## Rules
- One JSON object per ICO triplet with at least one reported numeric value; omit triplets with no reported values.
- Use plain numbers (no percent signs, no units).
- Do not invent or derive numbers; use only what is explicitly stated in Abstract/Results.
- Output raw JSON only (no markdown fencing, no extra text).
