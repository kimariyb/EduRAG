# Unified MLX RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate and quality-filter the RAG test dataset, then evaluate the real RAG system with four reasoned 1-5 Ragas metrics using `qwen3.5:4b-mlx` for every LLM role.

**Architecture:** Keep one typed model setting in `config.yaml`, make agent responses robust to MLX structured-output limitations, expose an immutable RAG answer trace, and implement all evaluation orchestration in `eval/rag.py`. Ragas receives a standard `EvaluationDataset` and custom `SingleTurnMetric` objects; each metric calls the shared configured LLM and stores both reason and score.

**Tech Stack:** Python 3.10, Ollama, qwen3.5:4b-mlx, LangChain Core, Ragas 0.3.9, PyMilvus, pytest, Loguru.

## Global Constraints

- Activate the environment with `source /opt/miniconda3/etc/profile.d/conda.sh` and `conda activate edurag` before every Python or pytest command.
- Use `qwen3.5:4b-mlx` for generation, critique, RAG answers, and Ragas judging.
- Keep prompts and logs in English; generated QA text and judge reasons are Simplified Chinese.
- Keep RAG routing, retrieval, and answer semantics unchanged.
- Keep PyTorch devices configuration-driven with `mps` in `config.yaml`.
- Replace existing generated data without a backup.
- Run Ragas with one worker on the 16 GB Apple Silicon host.

---

### Task 1: Verify Ragas compatibility and configure unified outputs

**Files:**
- Modify: `base/config.py`
- Modify: `config.yaml`
- Modify: `tests/test_base_config.py`

**Interfaces:**
- Produces: `EvalConfig.rag_predictions_path: str`
- Produces: `EvalConfig.rag_evaluation_path: str`
- Produces: `EvalConfig.rag_summary_path: str`
- Produces: `EvalConfig.ragas_max_workers: int`
- Produces: `EvalConfig.ragas_timeout: int`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_config_uses_unified_mlx_model_and_rag_evaluation_paths():
    config = load_config()
    assert config.llm.model == "qwen3.5:4b-mlx"
    assert Path(config.eval.rag_predictions_path).is_absolute()
    assert Path(config.eval.rag_evaluation_path).is_absolute()
    assert Path(config.eval.rag_summary_path).is_absolute()
    assert config.eval.ragas_max_workers == 1
    assert config.eval.ragas_timeout > 0
```

- [ ] **Step 2: Run the test and verify the expected attribute/model failure**

Run: `PYTHONPATH=. pytest -q tests/test_base_config.py::test_config_uses_unified_mlx_model_and_rag_evaluation_paths`

Expected: FAIL because the model is still `qwen3.5:2b` and the new fields do not exist.

- [ ] **Step 3: Add typed fields, path resolution, validation, and YAML values**

```python
@dataclass(frozen=True)
class EvalConfig:
    quality_threshold: int = 4
    critique_max_retries: int = 3
    test_samples_path: str = "eval/data/test_samples.jsonl"
    critique_results_path: str = "eval/data/test_samples_critiqued.jsonl"
    filtered_samples_path: str = "eval/data/test_samples_filtered.jsonl"
    rag_predictions_path: str = "eval/data/rag_predictions.jsonl"
    rag_evaluation_path: str = "eval/data/rag_evaluation.jsonl"
    rag_summary_path: str = "eval/data/rag_evaluation_summary.json"
    ragas_max_workers: int = 1
    ragas_timeout: int = 180
```

Set `llm.model: qwen3.5:4b-mlx`, resolve all three new paths relative to the
configuration file, parse both integers, and reject non-positive values.

- [ ] **Step 4: Run configuration tests**

Run: `PYTHONPATH=. pytest -q tests/test_base_config.py`

Expected: PASS.

### Task 2: Parse marker-based sample and critique responses

**Files:**
- Modify: `eval/datasets.py`
- Modify: `tests/test_eval_datasets.py`

**Interfaces:**
- Produces: `TestSampleAgent._parse_response(response: str) -> list[Mapping[str, Any]]`
- Produces: `CritiqueAgent._parse_response(response: str) -> CritiqueEvaluation`

- [ ] **Step 1: Write failing marker parsing and prompt-order tests**

```python
def test_test_sample_agent_parses_labeled_blocks():
    response = "QUESTION::: 什么是N-gram？\nANSWER::: 固定窗口语言模型。"
    assert SampleAgent._parse_response(response) == [
        {"question": "什么是N-gram？", "answer": "固定窗口语言模型。"}
    ]

def test_critique_prompt_places_each_reason_before_score():
    prompt = datasets_module.CritiqueAgent._build_prompt(sample())
    assert prompt.index("ANSWERABILITY_REASON:::") < prompt.index("ANSWERABILITY_SCORE:::")
```

- [ ] **Step 2: Run focused tests and verify parsing/order failures**

Run: `PYTHONPATH=. pytest -q tests/test_eval_datasets.py -k 'labeled_blocks or reason_before_score'`

Expected: FAIL because only JSON is accepted and the prompt uses JSON fields.

- [ ] **Step 3: Implement strict marker parsing with JSON compatibility**

Generation blocks use exactly `QUESTION:::` followed by `ANSWER:::`. Critique
blocks use `ANSWERABILITY_REASON:::`, `ANSWERABILITY_SCORE:::`,
`RELEVANCE_REASON:::`, `RELEVANCE_SCORE:::`, `STANDALONE_REASON:::`, and
`STANDALONE_SCORE:::`. Reject missing, empty, duplicate, non-integer, or
out-of-range fields. Keep `_decode_json_response` as a compatibility fallback.

- [ ] **Step 4: Run all dataset tests**

Run: `PYTHONPATH=. pytest -q tests/test_eval_datasets.py`

Expected: PASS.

### Task 3: Expose one-pass RAG answer traces

**Files:**
- Modify: `core/rag/system.py`
- Modify: `tests/test_rag_core.py`

**Interfaces:**
- Produces: `RAGAnswer` frozen dataclass with `answer`, `category`, `strategy`, and `documents`
- Produces: `RAGSystem.generate_answer_with_trace(query: str, source_filter: str | None = None) -> RAGAnswer`
- Preserves: `RAGSystem.generate_answer(...) -> str`

- [ ] **Step 1: Write a failing trace test**

```python
def test_rag_system_returns_answer_with_actual_retrieval_trace():
    system = RAGSystem(
        FakeVectorStore(),
        lambda prompt: "answer",
        query_classifier=StaticQueryClassifier(PROFESSIONAL_CONSULTATION_CATEGORY),
        strategy_selector=StaticStrategySelector(DIRECT_RETRIEVAL_STRATEGY),
    )
    result = system.generate_answer_with_trace("课程问题")
    assert result.answer == "answer"
    assert result.category == PROFESSIONAL_CONSULTATION_CATEGORY
    assert result.strategy == DIRECT_RETRIEVAL_STRATEGY
    assert [doc.page_content for doc in result.documents] == ["context"]
```

- [ ] **Step 2: Run the focused test and verify the missing-method failure**

Run: `PYTHONPATH=. pytest -q tests/test_rag_core.py::test_rag_system_returns_answer_with_actual_retrieval_trace`

Expected: FAIL with `AttributeError` for `generate_answer_with_trace`.

- [ ] **Step 3: Refactor preparation and generation without duplicate retrieval**

Add immutable internal preparation data, retain normalized category/strategy,
and return `RAGAnswer`. Make `generate_answer()` return
`generate_answer_with_trace(...).answer`. Preserve fallback and logging paths.

- [ ] **Step 4: Run RAG tests**

Run: `PYTHONPATH=. pytest -q tests/test_rag_core.py`

Expected: PASS.

### Task 4: Implement Ragas rubrics and score parsing in eval/rag.py

**Files:**
- Modify: `eval/rag.py`
- Create: `tests/test_eval_rag.py`

**Interfaces:**
- Produces: `RubricScore(reason: str, score: int)`
- Produces: `EvaluationRubric(name: str, required_columns: frozenset[str], instruction: str, anchors: Mapping[int, str])`
- Produces: `RagasRubricMetric(SingleTurnMetric)`
- Produces: `build_metrics(llm: Callable[[str], str]) -> list[RagasRubricMetric]`

- [ ] **Step 1: Write failing rubric score, prompt, and metric tests**

```python
def test_rubric_prompt_has_all_anchors_and_reason_before_score():
    metric = build_metrics(lambda prompt: "REASON::: 清晰。\nSCORE::: 5")[0]
    prompt = metric.build_prompt(sample())
    for score in range(1, 6):
        assert f"{score}:" in prompt
    assert prompt.index("REASON:::") < prompt.index("SCORE:::")

def test_parse_rubric_score_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="between 1 and 5"):
        parse_rubric_score("REASON::: bad\nSCORE::: 6")
```

- [ ] **Step 2: Run tests and verify imports/functions are missing**

Run: `PYTHONPATH=. pytest -q tests/test_eval_rag.py -k 'rubric'`

Expected: FAIL during import because the rubric types do not exist.

- [ ] **Step 3: Implement four complete English rubrics and robust parsing**

Every prompt includes question and only the metric-relevant fields, treats
them as untrusted data, lists concrete 1-5 anchors, asks for a concise Chinese
reason, and requires `REASON:::` before `SCORE:::`. The metric stores the
validated `RubricScore` by a deterministic sample key and returns the score as
`float` to Ragas.

- [ ] **Step 4: Run rubric tests**

Run: `PYTHONPATH=. pytest -q tests/test_eval_rag.py -k 'rubric or metric'`

Expected: PASS.

### Task 5: Implement prediction, Ragas execution, and reports in eval/rag.py

**Files:**
- Modify: `eval/rag.py`
- Modify: `tests/test_eval_rag.py`

**Interfaces:**
- Produces: `RAGPrediction.from_sample(sample: TestSample, result: RAGAnswer) -> RAGPrediction`
- Produces: `generate_predictions(samples, rag_system, output_path) -> list[RAGPrediction]`
- Produces: `build_evaluation_dataset(predictions) -> EvaluationDataset`
- Produces: `evaluate_predictions(predictions, metrics, config) -> list[RAGEvaluationRecord]`
- Produces: `summarize_evaluations(records) -> dict[str, Any]`
- Produces: `main() -> None`

- [ ] **Step 1: Write failing workflow tests**

```python
def test_build_evaluation_dataset_uses_actual_contexts():
    dataset = build_evaluation_dataset([prediction()])
    row = dataset.to_list()[0]
    assert row["user_input"] == "question"
    assert row["retrieved_contexts"] == ["retrieved"]
    assert row["response"] == "response"
    assert row["reference"] == "reference"

def test_summary_averages_only_completed_records():
    summary = summarize_evaluations([completed_record(), failed_record()])
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["metrics"]["faithfulness"]["mean"] == 4.0
```

- [ ] **Step 2: Run workflow tests and verify missing-function failures**

Run: `PYTHONPATH=. pytest -q tests/test_eval_rag.py -k 'dataset or prediction or summary or main'`

Expected: FAIL because the workflow functions do not exist.

- [ ] **Step 3: Implement resumable prediction and Ragas reporting**

Load only the configured filtered dataset. Reuse successful prediction rows by
normalized question. Call `ragas.evaluate()` with an `EvaluationDataset`, the
four custom metrics, `RunConfig(max_workers=1, timeout=config.eval.ragas_timeout)`,
`raise_exceptions=True`, and `show_progress=True`. Merge each Ragas score with
the reason stored by its metric, atomically write evaluation JSONL in source
order, and write a UTF-8 indented summary JSON.

- [ ] **Step 4: Implement main wiring**

`main()` loads config, validates the configured model name, initializes the
query classifier and populated vector store exactly as the interactive RAG
workflow does, generates/resumes predictions, evaluates them, and logs English
counts and output paths.

- [ ] **Step 5: Run eval/rag.py tests**

Run: `PYTHONPATH=. pytest -q tests/test_eval_rag.py`

Expected: PASS.

### Task 6: Verify code before the long local run

**Files:**
- Verify: all modified Python and YAML files

**Interfaces:**
- Consumes all interfaces from Tasks 1-5.

- [ ] **Step 1: Verify Ragas imports in edurag**

Run: `python -c "import ragas; print(ragas.__version__)"`

Expected: exit 0 and `0.3.9`. If the installed LangChain Community version
still lacks `chat_models.vertexai`, install the newest 0.3.x Community version
that provides that module, then rerun this command and the full tests.

- [ ] **Step 2: Run focused tests**

Run: `PYTHONPATH=. pytest -q tests/test_base_config.py tests/test_eval_datasets.py tests/test_eval_rag.py tests/test_rag_core.py`

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `PYTHONPATH=. pytest -q`

Expected: PASS with no new warnings.

### Task 7: Regenerate, critique, and evaluate with qwen3.5:4b-mlx

**Files:**
- Replace: `eval/data/test_samples.jsonl`
- Replace: `eval/data/test_samples_critiqued.jsonl`
- Replace: `eval/data/test_samples_filtered.jsonl`
- Create: `eval/data/rag_predictions.jsonl`
- Create: `eval/data/rag_evaluation.jsonl`
- Create: `eval/data/rag_evaluation_summary.json`

**Interfaces:**
- Consumes: `python -m eval.datasets`
- Consumes: `python -m eval.rag`

- [ ] **Step 1: Remove old 2B-generated JSONL outputs**

Remove the three configured sample files and any partial RAG evaluation files.
The user explicitly requested replacement without backup.

- [ ] **Step 2: Generate and critique at least 200 samples**

Run: `PYTHONPATH=. python -m eval.datasets`

Expected: 200 valid raw JSONL rows, 200 critique rows, and at least one passing
filtered row; logs identify `qwen3.5:4b-mlx`.

- [ ] **Step 3: Run end-to-end RAG evaluation**

Run: `PYTHONPATH=. python -m eval.rag`

Expected: one prediction and evaluation row per filtered sample and a summary
with four means between 1 and 5.

- [ ] **Step 4: Validate generated artifacts independently**

Parse every non-empty line with `json.loads`, assert raw and critique counts are
at least 200, assert filtered/prediction/evaluation counts match, assert every
metric has a non-empty reason and integer score from 1 to 5, and recompute the
summary means from the evaluation rows.

- [ ] **Step 5: Run final full tests**

Run: `PYTHONPATH=. pytest -q`

Expected: PASS.
