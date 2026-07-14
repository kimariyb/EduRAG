DIRECT_RETRIEVAL_STRATEGY = "direct_retrieval"
HYDE_RETRIEVAL_STRATEGY = "hyde_retrieval"
SUBQUERY_RETRIEVAL_STRATEGY = "subquery_retrieval"
BACKTRACKING_RETRIEVAL_STRATEGY = "backtracking_retrieval"

RETRIEVAL_STRATEGIES = (
    DIRECT_RETRIEVAL_STRATEGY,
    HYDE_RETRIEVAL_STRATEGY,
    SUBQUERY_RETRIEVAL_STRATEGY,
    BACKTRACKING_RETRIEVAL_STRATEGY,
)

RETRIEVAL_STRATEGY_LOG_NAMES = {
    DIRECT_RETRIEVAL_STRATEGY: "direct",
    HYDE_RETRIEVAL_STRATEGY: "hyde",
    SUBQUERY_RETRIEVAL_STRATEGY: "subquery",
    BACKTRACKING_RETRIEVAL_STRATEGY: "backtracking",
}

RETRIEVAL_STRATEGY_ALIASES = {
    "direct retrieval": DIRECT_RETRIEVAL_STRATEGY,
    "hyde retrieval": HYDE_RETRIEVAL_STRATEGY,
    "subquery retrieval": SUBQUERY_RETRIEVAL_STRATEGY,
    "backtracking retrieval": BACKTRACKING_RETRIEVAL_STRATEGY,
    "直接检索": DIRECT_RETRIEVAL_STRATEGY,
    "假设问题检索": HYDE_RETRIEVAL_STRATEGY,
    "子查询检索": SUBQUERY_RETRIEVAL_STRATEGY,
    "回溯问题检索": BACKTRACKING_RETRIEVAL_STRATEGY,
}


def normalize_retrieval_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower()
    for candidate in RETRIEVAL_STRATEGIES:
        if candidate in normalized:
            return candidate
    for alias, candidate in RETRIEVAL_STRATEGY_ALIASES.items():
        if alias in normalized:
            return candidate
    return DIRECT_RETRIEVAL_STRATEGY


GENERAL_KNOWLEDGE_CATEGORY = "general_knowledge"
PROFESSIONAL_CONSULTATION_CATEGORY = "professional_consultation"
QUERY_LABEL_MAP = {
    GENERAL_KNOWLEDGE_CATEGORY: 0,
    PROFESSIONAL_CONSULTATION_CATEGORY: 1,
}
QUERY_CATEGORY_ALIASES = {
    "通用知识": GENERAL_KNOWLEDGE_CATEGORY,
    "专业咨询": PROFESSIONAL_CONSULTATION_CATEGORY,
}
QUERY_CATEGORY_LOG_NAMES = {
    GENERAL_KNOWLEDGE_CATEGORY: "general_knowledge",
    PROFESSIONAL_CONSULTATION_CATEGORY: "professional_consultation",
}


def normalize_query_category(category: str) -> str:
    normalized = category.strip().lower()
    if normalized in QUERY_LABEL_MAP:
        return normalized
    return QUERY_CATEGORY_ALIASES.get(category.strip(), normalized)

DEFAULT_QUERY_MODEL = "bert-base-chinese"
DEFAULT_QUERY_MODEL_PATH = "bert_query_classifier"
DEFAULT_QUERY_TRAINING_DATA_PATH = "data/finetuning_data.jsonl"
DEFAULT_EMBEDDING_MODEL_PATH = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL_PATH = "./bge/bge-reranker-large"
DEFAULT_MODEL_DEVICE = "cpu"
DEFAULT_SEGMENTER_DEVICE = "cpu"
DEFAULT_LLM_MODEL = "qwen3.5:2b"
DEFAULT_CUSTOMER_SERVICE_PHONE = "400-000-0000"

DEFAULT_PARENT_CHUNK_SIZE = 1000
DEFAULT_CHILD_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 0.25
DEFAULT_RETRIEVAL_K = 10
DEFAULT_CANDIDATE_M = 3

DENSE_SEARCH_WEIGHT = 1.0
SPARSE_SEARCH_WEIGHT = 0.7
