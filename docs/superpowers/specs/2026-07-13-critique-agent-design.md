# Critique Agent Design

## Goal

Evaluate all generated RAG test samples with the local Ollama model and remove
samples that fail any quality criterion. The original generated dataset remains
unchanged.

## Scope

The evaluation covers exactly three criteria:

1. **Answerability**: the question can be answered clearly and unambiguously
   from its context.
2. **Domain relevance**: the question is useful to users learning or working
   with AI, machine learning, natural language processing, software
   development, or related IT education and training.
3. **Standalone quality**: the question is understandable without access to
   the source context and does not contain implicit document references.

Every criterion uses an integer score from 1 to 5. A sample passes only when
all three scores are greater than or equal to the configured threshold, which
defaults to 4.

## Architecture

`eval/datasets.py` will contain a single `CritiqueAgent`. It sends one prompt
per sample to Ollama and requests all three evaluations in one JSON object.
This keeps the criteria explicit while limiting the full 200-sample review to
approximately 200 model calls.

The prompt is written in English to match the project's prompt convention. It
asks the model to write concise evaluation reasons in Simplified Chinese and
return JSON only. The expected response has this shape:

```json
{
  "answerability": {"evaluation": "...", "score": 5},
  "relevance": {"evaluation": "...", "score": 4},
  "standalone": {"evaluation": "...", "score": 5}
}
```

Typed data classes validate that every criterion exists, each explanation is a
non-empty string, and each score is an integer in the range 1 through 5.

## Data Flow

1. Load `eval/data/test_samples.jsonl`.
2. Load the existing critique checkpoint, if present.
3. Skip samples already evaluated, using normalized questions as stable keys.
4. Send each remaining sample to `CritiqueAgent`.
5. Retry malformed responses or model failures up to the configured limit.
6. Append every completed result immediately to the critique checkpoint.
7. Rebuild the filtered dataset from passing critique records after evaluation.

The original test sample file is never modified. Full records are written to
`eval/data/test_samples_critiqued.jsonl`; accepted original samples are written
to `eval/data/test_samples_filtered.jsonl`.

Each full critique record contains the original four sample fields, the three
structured evaluations, a `passed` boolean, and `rejection_reasons`. Rejection
reasons identify criteria below the threshold. If all retry attempts fail, the
record is rejected with a `critique_agent_error` reason and the error is logged
in English.

A successful full record uses this shape:

```json
{
  "context": "...",
  "question": "...",
  "answer": "...",
  "source_doc": "...",
  "critiques": {
    "answerability": {"evaluation": "...", "score": 5},
    "relevance": {"evaluation": "...", "score": 4},
    "standalone": {"evaluation": "...", "score": 5}
  },
  "passed": true,
  "rejection_reasons": [],
  "error": null
}
```

An exhausted evaluation stores `critiques` as `null`, sets `passed` to
`false`, sets `rejection_reasons` to `["critique_agent_error"]`, and stores the
last exception message in `error`. The filtered file contains only the four
original sample fields so it remains directly consumable by the existing RAG
evaluation code.

## Configuration

A new `eval` section in `config.yaml` controls:

- `quality_threshold`, default `4`;
- `critique_max_retries`, default `3`;
- input test sample path;
- full critique result path;
- filtered sample path.

These values are represented by a typed `EvalConfig`, support the existing
`EDURAG_<SECTION>_<FIELD>` environment override convention, and resolve relative
paths against the configuration file directory.

## Error Handling

- Markdown-fenced JSON is accepted, but other non-JSON output is rejected.
- Missing criteria, empty explanations, boolean scores, non-integer scores, and
  scores outside 1 through 5 are invalid.
- Invalid model responses are retried without writing partial records.
- Exhausted retries produce a rejected error record so one bad response cannot
  stop the remaining 200-sample review.
- Invalid checkpoint rows are skipped with an English warning.

## Testing and Acceptance

Tests will cover prompt construction, valid and invalid response parsing,
threshold decisions, retry behavior, failed-review handling, checkpoint resume,
filtered output, configuration loading, and command-line workflow wiring.

After implementation, the project will call the configured local Ollama model
to review the existing 200 samples. Acceptance requires:

- exactly 200 full critique records corresponding to the 200 source samples;
- no duplicate reviewed questions;
- every accepted sample has all three scores at least 4;
- the filtered file contains only accepted original samples;
- all automated tests pass.

No RAG generation, retrieval, answering, or unrelated project behavior is
changed.
