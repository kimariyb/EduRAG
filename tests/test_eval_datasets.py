import json

import pytest
from langchain_core.documents import Document

import eval.datasets as datasets_module
from eval.datasets import TestSample as Sample
from eval.datasets import TestSampleAgent as SampleAgent
from eval.datasets import (
    generate_test_samples,
    load_test_samples,
    normalize_question,
)


def make_sample(index, *, question=None):
    return Sample(
        context=f"context {index}",
        question=question or f"问题 {index}？",
        answer=f"答案 {index}。",
        source_doc="knowledge.md",
    )


def make_evaluation(answerability=5, relevance=5, standalone=5):
    return datasets_module.CritiqueEvaluation(
        answerability=datasets_module.CritiqueScore(
            "上下文可以明确回答。",
            answerability,
        ),
        relevance=datasets_module.CritiqueScore(
            "问题与目标领域相关。",
            relevance,
        ),
        standalone=datasets_module.CritiqueScore(
            "问题可以独立理解。",
            standalone,
        ),
    )


def write_samples(path, samples):
    path.write_text(
        "".join(
            json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )


def test_test_sample_normalizes_and_serializes_required_fields():
    sample = Sample(
        context="  knowledge context  ",
        question="  什么是大语言模型？  ",
        answer="  一种使用大量数据训练的语言模型。  ",
        source_doc="  knowledge.md  ",
    )

    assert sample.to_dict() == {
        "context": "knowledge context",
        "question": "什么是大语言模型？",
        "answer": "一种使用大量数据训练的语言模型。",
        "source_doc": "knowledge.md",
    }
    assert Sample.from_dict(sample.to_dict()) == sample


@pytest.mark.parametrize(
    "field_name",
    ["context", "question", "answer", "source_doc"],
)
def test_test_sample_rejects_empty_required_fields(field_name):
    values = {
        "context": "context",
        "question": "question",
        "answer": "answer",
        "source_doc": "source.md",
    }
    values[field_name] = "   "

    with pytest.raises(ValueError, match=f"{field_name} cannot be empty"):
        Sample(**values)


def test_test_sample_agent_parses_fenced_json_array():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return """```json
[
  {
    "question": "课程包含哪些主要模块？",
    "answer": "包含基础和项目模块。"
  },
  {
    "question": "课程项目采用什么形式？",
    "answer": "采用企业实战项目。"
  }
]
```"""

    agent = SampleAgent(llm, samples_per_context=2)
    document = Document(
        page_content="课程包含基础、进阶和企业实战项目。",
        metadata={"file_path": "courses.md"},
    )

    samples = agent.generate(document)

    assert [sample.question for sample in samples] == [
        "课程包含哪些主要模块？",
        "课程项目采用什么形式？",
    ]
    assert all(sample.context == document.page_content for sample in samples)
    assert all(sample.source_doc == "courses.md" for sample in samples)
    assert "Simplified Chinese" in calls[0]
    assert "exactly 2" in calls[0]
    assert document.page_content in calls[0]
    assert "courses.md" not in calls[0]


def test_test_sample_agent_accepts_single_json_object():
    response = json.dumps(
        {"question": "课程持续多久？", "answer": "课程持续六个月。"},
        ensure_ascii=False,
    )
    agent = SampleAgent(lambda prompt: response, samples_per_context=1)
    document = Document(
        page_content="课程持续六个月。",
        metadata={"file_path": "course.md"},
    )

    assert agent.generate(document)[0].answer == "课程持续六个月。"


def test_test_sample_agent_parses_labeled_blocks():
    response = """
QUESTION::: 什么是N-gram？
ANSWER::: N-gram是对固定长度文本窗口进行建模的语言模型。

QUESTION::: N-gram的一个主要局限是什么？
ANSWER::: 参数空间会随着上下文长度快速增长。
""".strip()

    assert SampleAgent._parse_response(response) == [
        {
            "question": "什么是N-gram？",
            "answer": "N-gram是对固定长度文本窗口进行建模的语言模型。",
        },
        {
            "question": "N-gram的一个主要局限是什么？",
            "answer": "参数空间会随着上下文长度快速增长。",
        },
    ]


def test_test_sample_agent_rejects_malformed_json():
    agent = SampleAgent(lambda prompt: "not-json")
    document = Document(
        page_content="valid context",
        metadata={"file_path": "source.md"},
    )

    with pytest.raises(ValueError, match="valid JSON"):
        agent.generate(document)


def test_critique_agent_evaluates_all_criteria_in_one_call():
    calls = []
    response = json.dumps(
        {
            "answerability": {
                "evaluation": "上下文可以明确回答该问题。",
                "score": 5,
            },
            "relevance": {
                "evaluation": "该问题对机器学习学习者有用。",
                "score": 4,
            },
            "standalone": {
                "evaluation": "该问题脱离上下文后仍可理解。",
                "score": 5,
            },
        },
        ensure_ascii=False,
    )
    sample = make_sample(1, question="什么是语言模型？")
    agent = datasets_module.CritiqueAgent(
        lambda prompt: calls.append(prompt) or f"```json\n{response}\n```"
    )

    evaluation = agent.evaluate(sample)

    assert len(calls) == 1
    assert evaluation.scores == {
        "answerability": 5,
        "relevance": 4,
        "standalone": 5,
    }
    assert evaluation.relevance.evaluation == (
        "该问题对机器学习学习者有用。"
    )
    assert "Answerability" in calls[0]
    assert "Domain relevance" in calls[0]
    assert "Standalone quality" in calls[0]
    assert "Standalone score anchors" in calls[0]
    assert "You MUST assign 5" in calls[0]
    assert "Simplified Chinese" in calls[0]
    assert sample.question in calls[0]
    assert sample.context in calls[0]
    assert sample.source_doc not in calls[0]


def test_critique_agent_parses_reason_before_score_markers():
    response = """
ANSWERABILITY_REASON::: 上下文直接给出了定义。
ANSWERABILITY_SCORE::: 5
RELEVANCE_REASON::: 问题适合人工智能学习者。
RELEVANCE_SCORE::: 5
STANDALONE_REASON::: 问题明确指出了N-gram这一主题。
STANDALONE_SCORE::: 5
""".strip()
    sample = make_sample(1, question="什么是N-gram？")

    evaluation = datasets_module.CritiqueAgent(
        lambda prompt: response
    ).evaluate(sample)

    assert evaluation.scores == {
        "answerability": 5,
        "relevance": 5,
        "standalone": 5,
    }


def test_critique_prompt_places_each_reason_before_its_score():
    prompt = datasets_module.CritiqueAgent._build_prompt(make_sample(1))

    for criterion in ("ANSWERABILITY", "RELEVANCE", "STANDALONE"):
        assert prompt.index(f"{criterion}_REASON:::") < prompt.index(
            f"{criterion}_SCORE:::"
        )


def test_evaluation_agent_prompt_instructions_are_english():
    sample = Sample(
        context="An N-gram models a fixed-length token window.",
        question="What does an N-gram model?",
        answer="A fixed-length token window.",
        source_doc="knowledge.md",
    )
    prompts = [
        SampleAgent(lambda prompt: prompt)._build_prompt(sample.context),
        datasets_module.CritiqueAgent._build_prompt(sample),
    ]

    for prompt in prompts:
        assert not any("\u4e00" <= character <= "\u9fff" for character in prompt)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {
                "answerability": {"evaluation": "清晰。", "score": 5},
                "relevance": {"evaluation": "相关。", "score": 5},
            },
            "standalone",
        ),
        (
            {
                "answerability": {"evaluation": "清晰。", "score": True},
                "relevance": {"evaluation": "相关。", "score": 5},
                "standalone": {"evaluation": "独立。", "score": 5},
            },
            "integer",
        ),
        (
            {
                "answerability": {"evaluation": "清晰。", "score": 4.5},
                "relevance": {"evaluation": "相关。", "score": 5},
                "standalone": {"evaluation": "独立。", "score": 5},
            },
            "integer",
        ),
        (
            {
                "answerability": {"evaluation": "清晰。", "score": 6},
                "relevance": {"evaluation": "相关。", "score": 5},
                "standalone": {"evaluation": "独立。", "score": 5},
            },
            "between 1 and 5",
        ),
        (
            {
                "answerability": {"evaluation": "  ", "score": 5},
                "relevance": {"evaluation": "相关。", "score": 5},
                "standalone": {"evaluation": "独立。", "score": 5},
            },
            "evaluation cannot be empty",
        ),
    ],
)
def test_critique_agent_rejects_invalid_criterion(response, message):
    agent = datasets_module.CritiqueAgent(
        lambda prompt: json.dumps(response, ensure_ascii=False)
    )

    with pytest.raises(ValueError, match=message):
        agent.evaluate(make_sample(1))


def test_critique_evaluation_round_trips_dict():
    evaluation = datasets_module.CritiqueEvaluation.from_dict(
        {
            "answerability": {"evaluation": "清晰。", "score": 5},
            "relevance": {"evaluation": "相关。", "score": 4},
            "standalone": {"evaluation": "独立。", "score": 5},
        }
    )

    assert datasets_module.CritiqueEvaluation.from_dict(
        evaluation.to_dict()
    ) == evaluation


def test_critique_agent_forces_explicit_context_reference_to_score_one():
    response = json.dumps(
        {
            "answerability": {"evaluation": "可回答。", "score": 5},
            "relevance": {"evaluation": "相关。", "score": 5},
            "standalone": {"evaluation": "可独立理解。", "score": 5},
        },
        ensure_ascii=False,
    )
    sample = make_sample(
        1,
        question="根据提供的信息，句子概率越大意味着什么？",
    )

    evaluation = datasets_module.CritiqueAgent(
        lambda prompt: response
    ).evaluate(sample)

    assert evaluation.standalone.score == 1
    assert "根据提供的信息" in evaluation.standalone.evaluation


def test_critique_record_passes_only_when_all_scores_reach_threshold():
    passing = datasets_module.CritiqueRecord.from_evaluation(
        make_sample(1),
        make_evaluation(answerability=4, relevance=4, standalone=4),
        threshold=4,
    )
    rejected = datasets_module.CritiqueRecord.from_evaluation(
        make_sample(2),
        make_evaluation(answerability=3, relevance=4, standalone=2),
        threshold=4,
    )

    assert passing.passed is True
    assert passing.rejection_reasons == ()
    assert rejected.passed is False
    assert rejected.rejection_reasons == (
        "answerability",
        "standalone",
    )


def test_critique_record_serializes_success_and_error_records():
    successful = datasets_module.CritiqueRecord.from_evaluation(
        make_sample(1),
        make_evaluation(),
        threshold=4,
    )
    failed = datasets_module.CritiqueRecord.from_error(
        make_sample(2),
        "model unavailable",
    )

    assert datasets_module.CritiqueRecord.from_dict(
        successful.to_dict()
    ) == successful
    assert datasets_module.CritiqueRecord.from_dict(
        failed.to_dict()
    ) == failed
    assert failed.to_dict()["critiques"] is None
    assert failed.rejection_reasons == ("critique_agent_error",)


def test_evaluate_test_samples_retries_then_checkpoints(tmp_path):
    class FlakyAgent:
        def __init__(self):
            self.calls = 0

        def evaluate(self, sample):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("invalid response")
            return make_evaluation()

    agent = FlakyAgent()
    results_path = tmp_path / "critiqued.jsonl"
    filtered_path = tmp_path / "filtered.jsonl"

    records = datasets_module.evaluate_test_samples(
        [make_sample(1)],
        agent,
        results_path=results_path,
        filtered_path=filtered_path,
        threshold=4,
        max_retries=2,
    )

    assert agent.calls == 2
    assert records[0].passed is True
    assert datasets_module.load_critique_records(results_path) == records
    assert load_test_samples(filtered_path) == [make_sample(1)]


def test_evaluate_test_samples_records_exhausted_agent_error(tmp_path):
    class FailingAgent:
        def __init__(self):
            self.calls = 0

        def evaluate(self, sample):
            self.calls += 1
            raise RuntimeError("Ollama is unavailable")

    agent = FailingAgent()
    filtered_path = tmp_path / "filtered.jsonl"

    records = datasets_module.evaluate_test_samples(
        [make_sample(1)],
        agent,
        results_path=tmp_path / "critiqued.jsonl",
        filtered_path=filtered_path,
        threshold=4,
        max_retries=2,
    )

    assert agent.calls == 2
    assert records[0].passed is False
    assert records[0].critiques is None
    assert records[0].error == "Ollama is unavailable"
    assert records[0].rejection_reasons == ("critique_agent_error",)
    assert load_test_samples(filtered_path) == []


def test_evaluate_test_samples_resumes_existing_records(tmp_path):
    sample = make_sample(1)
    record = datasets_module.CritiqueRecord.from_evaluation(
        sample,
        make_evaluation(relevance=4),
        threshold=4,
    )
    results_path = tmp_path / "critiqued.jsonl"
    results_path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class UnexpectedAgent:
        def evaluate(self, sample):
            raise AssertionError("existing sample must not be evaluated again")

    records = datasets_module.evaluate_test_samples(
        [sample],
        UnexpectedAgent(),
        results_path=results_path,
        filtered_path=tmp_path / "filtered.jsonl",
        threshold=4,
        max_retries=1,
    )

    assert records == [record]


def test_evaluate_test_samples_retries_existing_error_record(tmp_path):
    sample = make_sample(1)
    error_record = datasets_module.CritiqueRecord.from_error(
        sample,
        "temporary model error",
    )
    results_path = tmp_path / "critiqued.jsonl"
    results_path.write_text(
        json.dumps(error_record.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class RecoveredAgent:
        def __init__(self):
            self.calls = 0

        def evaluate(self, active_sample):
            self.calls += 1
            return make_evaluation()

    agent = RecoveredAgent()
    records = datasets_module.evaluate_test_samples(
        [sample],
        agent,
        results_path=results_path,
        filtered_path=tmp_path / "filtered.jsonl",
        threshold=4,
        max_retries=1,
    )

    assert agent.calls == 1
    assert records[0].passed is True
    assert records[0].error is None


def test_load_critique_records_skips_malformed_rows(tmp_path):
    record = datasets_module.CritiqueRecord.from_evaluation(
        make_sample(1),
        make_evaluation(),
        threshold=4,
    )
    results_path = tmp_path / "critiqued.jsonl"
    results_path.write_text(
        "not-json\n"
        + json.dumps(record.to_dict(), ensure_ascii=False)
        + "\n"
        + json.dumps({"question": "missing fields"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert datasets_module.load_critique_records(results_path) == [record]


def test_evaluate_test_samples_deduplicates_and_preserves_source_order(
    tmp_path,
):
    first = make_sample(1, question="第二个问题？")
    second = make_sample(2, question="第一个问题？")
    duplicate = make_sample(3, question="  第二个问题？  ")

    class RecordingAgent:
        def __init__(self):
            self.questions = []

        def evaluate(self, sample):
            self.questions.append(sample.question)
            return make_evaluation(relevance=3 if sample == second else 5)

    agent = RecordingAgent()
    filtered_path = tmp_path / "filtered.jsonl"
    records = datasets_module.evaluate_test_samples(
        [first, second, duplicate],
        agent,
        results_path=tmp_path / "critiqued.jsonl",
        filtered_path=filtered_path,
        threshold=4,
        max_retries=1,
    )

    assert agent.questions == [first.question, second.question]
    assert [record.sample for record in records] == [first, second]
    assert load_test_samples(filtered_path) == [first]
    filtered_row = json.loads(
        filtered_path.read_text(encoding="utf-8").strip()
    )
    assert set(filtered_row) == {
        "context",
        "question",
        "answer",
        "source_doc",
    }


def test_normalize_question_collapses_case_and_whitespace():
    assert normalize_question("  What   IS RAG?  ") == "what is rag?"


def test_load_test_samples_skips_malformed_checkpoint_rows(tmp_path):
    output_path = tmp_path / "samples.jsonl"
    valid_sample = make_sample(1)
    output_path.write_text(
        json.dumps(valid_sample.to_dict(), ensure_ascii=False)
        + "\nnot-json\n"
        + json.dumps({"question": "missing fields"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert load_test_samples(output_path) == [valid_sample]


def test_generate_test_samples_rejects_target_below_minimum(tmp_path):
    document = Document(
        page_content="context",
        metadata={"file_path": "source.md"},
    )
    agent = SampleAgent(lambda prompt: "[]")

    with pytest.raises(ValueError, match="at least 200"):
        generate_test_samples(
            [document],
            agent,
            output_path=tmp_path / "samples.jsonl",
            target_size=199,
        )


def test_generate_test_samples_rejects_empty_documents(tmp_path):
    agent = SampleAgent(lambda prompt: "[]")

    with pytest.raises(ValueError, match="documents cannot be empty"):
        generate_test_samples(
            [],
            agent,
            output_path=tmp_path / "samples.jsonl",
            target_size=200,
        )


def test_generate_test_samples_writes_exact_target(tmp_path):
    output_path = tmp_path / "samples.jsonl"
    documents = [
        Document(
            page_content=f"context {index}",
            metadata={"file_path": "knowledge.md"},
        )
        for index in range(100)
    ]

    class Agent:
        def generate(self, document):
            index = document.page_content.split()[-1]
            return [
                Sample(
                    context=document.page_content,
                    question=f"问题 {index}-{suffix}？",
                    answer=f"答案 {index}-{suffix}。",
                    source_doc="knowledge.md",
                )
                for suffix in ("a", "b")
            ]

    samples = generate_test_samples(
        documents,
        Agent(),
        output_path=output_path,
        target_size=200,
        seed=7,
    )

    assert len(samples) == 200
    assert len(load_test_samples(output_path)) == 200


def test_generate_test_samples_resumes_existing_checkpoint(tmp_path):
    output_path = tmp_path / "samples.jsonl"
    existing_samples = [make_sample(index) for index in range(199)]
    write_samples(output_path, existing_samples)
    calls = []

    class Agent:
        def generate(self, document):
            calls.append(document.page_content)
            return [make_sample(200)]

    samples = generate_test_samples(
        [Document(page_content="new context")],
        Agent(),
        output_path=output_path,
        target_size=200,
        max_attempts=1,
    )

    assert len(samples) == 200
    assert calls == ["new context"]
    assert len(load_test_samples(output_path)) == 200


def test_generate_test_samples_retries_failed_agent_call(tmp_path):
    output_path = tmp_path / "samples.jsonl"
    write_samples(output_path, [make_sample(index) for index in range(198)])
    calls = 0

    class Agent:
        def generate(self, document):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("invalid model response")
            return [make_sample(198), make_sample(199)]

    samples = generate_test_samples(
        [Document(page_content="context")],
        Agent(),
        output_path=output_path,
        target_size=200,
        max_attempts=2,
    )

    assert len(samples) == 200
    assert calls == 2


def test_generate_test_samples_skips_duplicate_questions(tmp_path):
    output_path = tmp_path / "samples.jsonl"
    existing_samples = [make_sample(index) for index in range(199)]
    write_samples(output_path, existing_samples)
    calls = 0

    class Agent:
        def generate(self, document):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [make_sample(999, question="  问题 0？  ")]
            return [make_sample(200)]

    samples = generate_test_samples(
        [Document(page_content="context")],
        Agent(),
        output_path=output_path,
        target_size=200,
        max_attempts=2,
    )

    assert len(samples) == 200
    assert calls == 2


def test_generate_test_samples_raises_when_attempt_budget_is_exhausted(
    tmp_path,
):
    output_path = tmp_path / "samples.jsonl"
    existing_samples = [make_sample(index) for index in range(199)]
    write_samples(output_path, existing_samples)

    class DuplicateAgent:
        def generate(self, document):
            return [make_sample(999, question="问题 0？")]

    with pytest.raises(
        RuntimeError,
        match="generated 199 of 200 test samples",
    ):
        generate_test_samples(
            [Document(page_content="context")],
            DuplicateAgent(),
            output_path=output_path,
            target_size=200,
            max_attempts=2,
        )


def test_main_builds_default_test_sample_and_critique_workflow(
    monkeypatch,
    tmp_path,
):
    knowledge_path = tmp_path / "knowledge"
    test_samples_path = tmp_path / "test_samples.jsonl"
    critique_results_path = tmp_path / "test_samples_critiqued.jsonl"
    filtered_samples_path = tmp_path / "test_samples_filtered.jsonl"
    config = type(
        "Config",
        (),
        {
            "rag": type(
                "RAGConfig",
                (),
                {"knowledge_base_path": str(knowledge_path)},
            )(),
            "eval": type(
                "EvalConfig",
                (),
                {
                    "quality_threshold": 4,
                    "critique_max_retries": 3,
                    "test_samples_path": str(test_samples_path),
                    "critique_results_path": str(critique_results_path),
                    "filtered_samples_path": str(filtered_samples_path),
                },
            )(),
        },
    )()
    documents = [Document(page_content="knowledge chunk")]
    calls = {}
    llm = object()
    sample_agent = object()
    critique_agent = object()
    samples = [make_sample(index) for index in range(200)]
    records = [
        datasets_module.CritiqueRecord.from_evaluation(
            sample,
            make_evaluation(),
            threshold=4,
        )
        for sample in samples
    ]

    monkeypatch.setattr(
        datasets_module,
        "load_config",
        lambda: config,
        raising=False,
    )
    monkeypatch.setattr(
        datasets_module,
        "parse_document_from_dir",
        lambda path, config=None: (
            calls.update(parse=(path, config)) or documents
        ),
        raising=False,
    )
    monkeypatch.setattr(
        datasets_module,
        "ChatLLM",
        lambda active_config: (
            calls.update(llm_config=active_config) or llm
        ),
        raising=False,
    )
    monkeypatch.setattr(
        datasets_module,
        "TestSampleAgent",
        lambda active_llm: (
            calls.update(sample_agent_llm=active_llm) or sample_agent
        ),
    )
    monkeypatch.setattr(
        datasets_module,
        "CritiqueAgent",
        lambda active_llm: (
            calls.update(critique_agent_llm=active_llm) or critique_agent
        ),
    )
    monkeypatch.setattr(
        datasets_module,
        "generate_test_samples",
        lambda docs, active_agent, **kwargs: (
            calls.update(
                generate=(docs, active_agent, kwargs)
            )
            or samples
        ),
    )
    monkeypatch.setattr(
        datasets_module,
        "evaluate_test_samples",
        lambda active_samples, active_agent, **kwargs: (
            calls.update(
                evaluate=(active_samples, active_agent, kwargs)
            )
            or records
        ),
    )

    datasets_module.main()

    assert calls["parse"] == (knowledge_path, config)
    assert calls["llm_config"] is config
    assert calls["sample_agent_llm"] is llm
    assert calls["critique_agent_llm"] is llm
    assert calls["generate"] == (
        documents,
        sample_agent,
        {
            "output_path": str(test_samples_path),
            "target_size": 200,
        },
    )
    assert calls["evaluate"] == (
        samples,
        critique_agent,
        {
            "results_path": str(critique_results_path),
            "filtered_path": str(filtered_samples_path),
            "threshold": 4,
            "max_retries": 3,
        },
    )
