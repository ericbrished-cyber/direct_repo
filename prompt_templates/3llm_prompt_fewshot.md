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

- use them exactly as written in the JSON output,
- never modify their wording,
- never expand, merge, split, or normalize them,
- never create ANY new ICO triplets,
- never change intervention, comparator, or outcome names,
- never infer additional outcomes or arms from the text.

If the study reports similar or related outcomes, ignore them unless they match an ICO triplet exactly in meaning.

---------------------------------------------------------------------
SEMANTIC MATCHING RULE (CRITICAL)

Although the ICO triplets in {ico_list} must be used *exactly as written* in the JSON output:

You MUST match interventions, comparators, and outcomes in the article *semantically*, not textually.

This means:

- The article may use synonyms, abbreviations, paraphrases, or group labels (“treatment arm”, “control group”, “usual care”, “placebo”, “FPG”, “blood glucose after fasting”, etc.).
- The intervention/comparator/outcome in {ico_list} may NOT appear verbatim in the text.
- You must identify equivalents based on clinical meaning, not wording.
- Do NOT require exact string matches.
- Use the {ico_list} names only in the output, not as literal search patterns.

Failing to find verbatim matches does NOT mean the ICO is absent.

---------------------------------------------------------------------
STRICT ITERATION LOGIC (ABSOLUTELY REQUIRED)

Treat {ico_list} as the complete and final set of ICO triplets.

For each ICO triplet (in order):

1. Search the article ONLY for information corresponding to that ICO (based on semantic meaning).

2. If at least one numeric value exists → produce exactly one JSON object for that triplet and populate the numeric fields.

3. If no numeric values exist → produce one JSON object with all numeric fields set to null.

4. Continuous outcomes:
   - explicitly search for group sizes, mean, and standard deviation for each arm,
   - fill them if reported,
   - null is allowed only if the PDF omits the value.

5. DO NOT generate any extraction for outcomes or arms not in {ico_list}.

6. The final JSON must contain AT MOST len({ico_list}) objects.

7. If zero objects are produced, still return: {"extractions": []}.

You may not scan the article for outcomes outside {ico_list}.

---------------------------------------------------------------------
EXTRACTION RULES

Continuous outcomes (when explicitly reported):
- intervention_group_size
- comparator_group_size
- intervention_mean
- comparator_mean
- intervention_standard_deviation
- comparator_standard_deviation

Continuous rows are complete only if group sizes, means, and SDs are filled when present.

Binary outcomes (when explicitly reported):
- intervention_group_size
- comparator_group_size
- intervention_events
- comparator_events
- numeric rates if reported

---------------------------------------------------------------------
ALLOWED INFERENCE (EVENTS FROM RATE × N)

If only rate (%) and group size are reported:

events = round((rate / 100) * group_size)

Rules:
- 0 ≤ events ≤ group_size
- Keep original rate in a *_rate field if present.
- Do NOT infer means, standard deviations, or group sizes.

---------------------------------------------------------------------
GENERAL RULES

- Use plain numbers (no %, no units).
- Missing values → null.
- Output must be raw valid JSON.
- No narrative text.
- No comments.

---------------------------------------------------------------------
ICO UNIQUENESS (CRITICAL)

Exactly one JSON extraction per ICO triplet.

If multiple values exist → choose the most precise or the value from the main Results section.

---------------------------------------------------------------------
TIMEPOINT RULE

- If the ICO triplet specifies a timepoint → use only that timepoint.
- If not:
  - prefer post-intervention values,
  - if multiple exist, choose the latest.

---------------------------------------------------------------------
FINAL RULE

Output NO ICO triplets other than those in {ico_list}.

---------------------------------------------------------------------
EXAMPLES (FEW-SHOT)

Below are example input texts and the corresponding correct JSON output
under the rules above. The outputs follow the exact ICO schema and use
only ICO triplets taken from the annotated gold standard for the allowed PMCIDs.

Each example shows:
- the raw input text the model sees, and
- the correct `{"extractions": [...]}` JSON it must return.

---------------------------------------------------------------------
EXAMPLE 1 – Continuous outcomes (pmcid = 1216327)

EXAMPLE 1 INPUT

"Mean body weight gain: WGJ +50.6 g vs WA -0.7 g (n=30 per arm). Duration of illness (h): AJ 49.4 ± 32.6, WGJ 47.5 ± 38.9, WA 26.5 ± 27.4 (n=30/arm). Fecal losses (g/kg/h): AJ 3.94 ± 2.35, WGJ 3.59 ± 2.35, WA 2.19 ± 1.63 (n=30/arm)."

EXAMPLE 1 OUTPUT

```json
{
  "extractions": [
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "1216327",
      "outcome": "The mean body weight gain",
      "intervention": "White Grape Juice (WGJ)",
      "comparator": "colored and flavored water (WA)",
      "outcome_type": "continuous",
      "intervention_events": null,
      "intervention_group_size": 30,
      "comparator_events": null,
      "comparator_group_size": 30,
      "intervention_mean": 50.6,
      "intervention_standard_deviation": null,
      "comparator_mean": -0.7,
      "comparator_standard_deviation": null
    },
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "1216327",
      "outcome": "The duration of the illness",
      "intervention": "White Grape Juice (WGJ)",
      "comparator": "colored and flavored water (WA)",
      "outcome_type": "continuous",
      "intervention_events": null,
      "intervention_group_size": 30,
      "comparator_events": null,
      "comparator_group_size": 30,
      "intervention_mean": 47.5,
      "intervention_standard_deviation": 38.9,
      "comparator_mean": 26.5,
      "comparator_standard_deviation": 27.4
    },
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "1216327",
      "outcome": "The fecal losses",
      "intervention": "White Grape Juice (WGJ)",
      "comparator": "colored and flavored water (WA)",
      "outcome_type": "continuous",
      "intervention_events": null,
      "intervention_group_size": 30,
      "comparator_events": null,
      "comparator_group_size": 30,
      "intervention_mean": 3.59,
      "intervention_standard_deviation": 2.35,
      "comparator_mean": 2.19,
      "comparator_standard_deviation": 1.63
    }
  ]
}
EXAMPLE 2 – Binary outcome with events and group sizes (TEAEs, pmcid = 5771543)

EXAMPLE 2 INPUT

"A total of 220/339 (64.9 %) patients experienced AEs during the on-therapy period (desvenlafaxine, 69/115 [60.0%]; fluoxetine, 72/112 [64.3%]; and placebo, 79/112 [70.5 %])."

EXAMPLE 2 OUTPUT
{
  "extractions": [
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "5771543",
      "outcome": "Treatment-emergent adverse events (TEAEs)",
      "intervention": "desvenlafaxine",
      "comparator": "placebo",
      "outcome_type": "binary",
      "intervention_events": 69,
      "intervention_group_size": 115,
      "comparator_events": 79,
      "comparator_group_size": 112,
      "intervention_mean": null,
      "intervention_standard_deviation": null,
      "comparator_mean": null,
      "comparator_standard_deviation": null
    },
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "5771543",
      "outcome": "Treatment-emergent adverse events (TEAEs)",
      "intervention": "fluoxetine",
      "comparator": "placebo",
      "outcome_type": "binary",
      "intervention_events": 72,
      "intervention_group_size": 112,
      "comparator_events": 79,
      "comparator_group_size": 112,
      "intervention_mean": null,
      "intervention_standard_deviation": null,
      "comparator_mean": null,
      "comparator_standard_deviation": null
    }
  ]
}
EXAMPLE 3 – Binary outcome with explicit timepoint (CGI-I, pmcid = 5771543)

EXAMPLE 3 INPUT

"At week 8, CGI-I response rates were 78.2% (79/101) for fluoxetine, 68.7% (68/99) for desvenlafaxine, and 62.6% (62/99) for placebo."

EXAMPLE 3 OUTPUT
{
  "extractions": [
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "5771543",
      "outcome": "CGI-I response rate",
      "intervention": "fluoxetine",
      "comparator": "placebo",
      "outcome_type": "binary",
      "intervention_events": 79,
      "intervention_group_size": 101,
      "comparator_events": 62,
      "comparator_group_size": 99,
      "intervention_mean": null,
      "intervention_standard_deviation": null,
      "comparator_mean": null,
      "comparator_standard_deviation": null
    },
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "5771543",
      "outcome": "CGI-I response rate",
      "intervention": "desvenlafaxine",
      "comparator": "placebo",
      "outcome_type": "binary",
      "intervention_events": 68,
      "intervention_group_size": 99,
      "comparator_events": 62,
      "comparator_group_size": 99,
      "intervention_mean": null,
      "intervention_standard_deviation": null,
      "comparator_mean": null,
      "comparator_standard_deviation": null
    }
  ]
}
EXAMPLE 4 – Continuous outcome with only group sizes reported (pmcid = 5244530)

This example illustrates a case where the ICO triplet exists, and group sizes are
known, but no continuous summary statistics (means or SDs) are reported. All
unknown numeric fields must be set to null.

EXAMPLE 4 INPUT

"Remission rate was evaluated at 12 months in both groups. The person-centred GP consultation group included 125 patients, and the treatment as usual (TAU) group included 133 patients, but no continuous summary statistics (means or standard deviations) for remission were reported."

EXAMPLE 4 OUTPUT
{
  "extractions": [
    {
      "id": null,
      "evidence_inference_prompt_id": null,
      "pmcid": "5244530",
      "outcome": "remission rate",
      "intervention": "person-centred general practitioners (GP) consultations",
      "comparator": "treatment as usual (TAU)",
      "outcome_type": "continuous",
      "intervention_events": null,
      "intervention_group_size": 125,
      "comparator_events": null,
      "comparator_group_size": 133,
      "intervention_mean": null,
      "intervention_standard_deviation": null,
      "comparator_mean": null,
      "comparator_standard_deviation": null
    }
  ]
}
