# Native Ragas Metrics Evaluation Design

## Objective

Replace the custom 1–5 RAG judge in `eval/rag.py` with the native Ragas
0.3.9 implementations of Context Precision, Context Recall, Answer
Relevancy, and Faithfulness. Execute the evaluation only on predictions
that actually used retrieved context.

## Scope

The existing RAG answer generation and trace collection remain unchanged.
The evaluation stage will reuse successful predictions from
`eval/data/rag_predictions.jsonl` and select records that meet all of these
conditions:

- the prediction has no error;
- the route is `professional_consultation`;
- at least one non-empty retrieved context is present.

General-knowledge predictions are excluded because the system deliberately
does not retrieve context for that route. Assigning context-based scores to
those records would produce misleading zeros or undefined values.

## Metric Implementation

The evaluation will instantiate these Ragas metric classes directly:

- `ContextPrecision`: determines whether relevant contexts are ranked ahead
  of irrelevant contexts, using the question, retrieved contexts, and
  reference answer.
- `ContextRecall`: determines how much of the reference answer can be
  attributed to the retrieved contexts.
- `ResponseRelevancy`: measures how directly the response addresses the
  question. It uses the evaluation LLM to generate reverse questions and
  BGE-M3 dense embeddings for semantic similarity.
- `Faithfulness`: measures whether claims in the response are supported by
  the retrieved contexts.

All metrics produce their native continuous scores, normally in the range
from 0 to 1. The custom `RubricScore`, custom judge prompts, and retained
1–5 rationales will be removed.

## Model Adapters

The Ragas LLM adapter will use the configured Ollama OpenAI-compatible
endpoint and the configured `qwen3.5:4b-mlx` model. This keeps the LLM used
for evaluation consistent with the model used for sample generation,
critique, and RAG answers.

Answer Relevancy will use the dense output of the existing configured
BGE-M3 embedding function. A small Ragas embedding adapter will expose
`embed_query` and `embed_documents` without loading a second embedding
model, which is important on the 16 GB Apple Silicon machine.

## Data Flow

1. Load the quality-filtered test samples and build the normal RAG system.
2. Resume or generate RAG predictions with their actual context traces.
3. Filter predictions to the eligible retrieval route.
4. Convert eligible predictions to an `EvaluationDataset` containing
   `user_input`, `retrieved_contexts`, `response`, and `reference`.
5. Evaluate one eligible sample at a time with the four native metrics.
6. Save each completed record as a checkpoint.
7. Write a summary containing input, eligible, excluded, completed, and
   failed counts plus the mean and count for each metric.

Evaluation remains single-worker to avoid concurrent model pressure on
unified memory. Existing retry and checkpoint behavior remains, but the
old evaluation JSONL and summary are replaced because their schemas are
incompatible with native floating-point metric results.

## Persistence Schema

Each evaluation JSONL record will retain the original prediction fields and
store metrics as numeric values:

```json
{
  "question": "How long is the training program?",
  "retrieved_contexts": ["The training program lasts six months."],
  "response": "The training program lasts six months.",
  "reference": "Six months.",
  "metrics": {
    "context_precision": 0.9,
    "context_recall": 0.8,
    "answer_relevancy": 0.85,
    "faithfulness": 1.0
  },
  "evaluation_error": null
}
```

Scores must be finite numeric values. Failed records contain an empty metric
mapping and a non-empty `evaluation_error`.

## Error Handling

- Reject predictions without actual retrieval before creating the Ragas
  dataset.
- Treat `NaN`, infinity, missing metrics, or non-numeric scores as evaluation
  failures.
- Retry transient LLM or output-parsing failures using the configured retry
  count.
- Preserve successful records on rerun and retry only failed or missing
  records.
- Keep all workflow logs in English.

## Testing and Execution

Tests will be written before production changes and will verify:

- route-aware prediction filtering;
- use of the four native Ragas metric classes;
- correct LLM and BGE-M3 adapter wiring;
- standard Ragas dataset columns;
- finite floating-point score validation and persistence;
- checkpoint resume behavior;
- aggregate counts and metric means;
- the complete `main()` orchestration.

After the focused tests and full project suite pass, the old evaluation
outputs will be removed and the workflow will run in the `edurag` Conda
environment against local Ollama. The generated JSONL and summary will then
be independently validated for record counts, score ranges, and recomputed
means.
