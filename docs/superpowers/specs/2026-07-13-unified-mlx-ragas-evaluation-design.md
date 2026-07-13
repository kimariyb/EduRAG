# Unified MLX RAG Evaluation Design

## Goal

Use `qwen3.5:4b-mlx` as the single Ollama model for test-sample
generation, sample critique, RAG response generation, and Ragas judging.
Regenerate the existing dataset without retaining the earlier 2B output and
evaluate the quality-filtered samples end to end in `eval/rag.py`.

## Constraints

- Run every Python command in the `edurag` Conda environment.
- Keep all prompts and log messages in English.
- Keep generated questions, reference answers, and judge reasons in
  Simplified Chinese.
- Preserve the RAG system's routing, retrieval, and answer behavior.
- Use the configured MPS devices for PyTorch models; do not hard-code devices.
- Use `qwen3.5:4b-mlx` for every LLM role.
- Use Ragas 0.3.9 for evaluation orchestration.
- Replace the existing generated, critiqued, and filtered JSONL files without
  creating backups.

## Architecture

### Configuration

The shared `llm.model` setting becomes `qwen3.5:4b-mlx`. `EvalConfig` gains
three resolved output paths for RAG predictions, per-sample evaluation
records, and the aggregate summary. It also gains sequential Ragas execution
settings so the 16 GB Apple Silicon machine does not load concurrent Ollama
requests.

### Stable agent output

The MLX model does not reliably obey JSON Schema. Agent prompts therefore use
explicit text markers. The Test Sample Agent emits repeated
`QUESTION:::`/`ANSWER:::` blocks, and the Critique Agent emits each reason
before its integer score. Application code validates those markers and owns
JSONL serialization. Existing JSON responses remain readable so checkpoint
loading and unit fixtures stay compatible.

### RAG tracing

`RAGSystem` exposes an immutable answer result containing the final answer,
normalized category, selected strategy, and actual retrieved documents.
`generate_answer()` delegates to this traced path, so interactive behavior is
unchanged while evaluation can observe the exact contexts used for an answer.

### Ragas evaluation

`eval/rag.py` owns the evaluation workflow. It loads the filtered JSONL,
executes the real RAG system once per sample, persists resumable prediction
records, converts them to a Ragas `EvaluationDataset`, and calls
`ragas.evaluate()` with four custom `SingleTurnMetric` implementations.

Each metric uses the same configured `ChatLLM` and an English rubric prompt.
The prompt requires `REASON:::` before `SCORE:::`. Scores are integers from 1
to 5, while reasons are retained in the per-sample JSONL report.

The metrics are:

1. `retrieval_relevance`: whether retrieved contexts directly help answer the
   question and avoid unrelated material.
2. `retrieval_completeness`: whether retrieved contexts contain the facts
   needed to reproduce the reference answer.
3. `faithfulness`: whether every material claim in the generated response is
   supported by retrieved contexts.
4. `answer_correctness`: whether the generated response is correct, complete,
   direct, and consistent with the reference answer.

Ragas runs with one worker. Metric failures retry through the shared judge
call policy and fail closed rather than silently assigning a good score.

## Persistence

- `eval/data/test_samples.jsonl`: regenerated raw samples.
- `eval/data/test_samples_critiqued.jsonl`: regenerated critique details.
- `eval/data/test_samples_filtered.jsonl`: regenerated passing samples.
- `eval/data/rag_predictions.jsonl`: RAG response and trace for each accepted
  sample.
- `eval/data/rag_evaluation.jsonl`: prediction plus four score/reason pairs.
- `eval/data/rag_evaluation_summary.json`: count, failure count, arithmetic
  mean per metric, and overall mean.

Each JSONL row is flushed after validation. Final files are rewritten in input
order so interrupted runs can resume without changing report ordering.

## Error handling

- Invalid model output is rejected with a descriptive English exception.
- Generation and critique retain bounded retries and checkpointing.
- RAG prediction failures are persisted as failures and excluded from metric
  scoring until retried successfully.
- Empty retrieved contexts are valid and receive low retrieval/faithfulness
  scores according to the rubrics; this preserves end-to-end routing truth.
- Ragas import compatibility is verified before the long-running workflow.

## Verification

Unit tests cover marker parsing, rubric anchors, RAG trace propagation,
Ragas dataset conversion, score/reason persistence, summaries, configuration,
and main wiring. The full project test suite is run before regenerating data.
After the real local run, all JSONL files are parsed, row counts are checked,
the configured model name is verified, and the summary is recomputed from the
per-sample records.
