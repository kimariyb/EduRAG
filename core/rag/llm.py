from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from base.config import AppConfig, load_config
from base.logger import logger


log = logger.bind(module=__name__)
DEFAULT_SYSTEM_PROMPT = (
    "You are a reliable education support assistant. Follow the user's "
    "request while respecting the constraints in the user prompt."
)


def create_openai_client(config: AppConfig) -> Any:
    try:
        from openai import DefaultHttpxClient, OpenAI
    except ImportError as exc:
        raise RuntimeError("the openai package is required for LLM calls") from exc

    if not config.llm.api_key:
        raise ValueError("llm.api_key is not configured")
    client_arguments: dict[str, Any] = {
        "api_key": config.llm.api_key,
        "base_url": config.llm.base_url,
    }
    if config.llm.provider == "ollama":
        client_arguments["http_client"] = DefaultHttpxClient(
            trust_env=False
        )
    client = OpenAI(**client_arguments)
    log.info(
        "Created OpenAI-compatible LLM client: provider={}, base_url={}",
        config.llm.provider,
        config.llm.base_url,
    )
    return client


class ChatLLM:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        client: Any | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.config = config or load_config()
        self.client = client
        self.system_prompt = system_prompt
        log.info("Chat LLM initialized: model={}", self.config.llm.model)

    def _completion_arguments(self, prompt: str) -> dict[str, Any]:
        """Build the shared OpenAI-compatible chat request arguments."""
        return {
            "model": self.config.llm.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "reasoning_effort": self.config.llm.reasoning_effort,
        }

    def _client(self) -> Any:
        if self.client is None:
            self.client = create_openai_client(self.config)
        return self.client

    def __call__(self, prompt: str) -> str:
        completion = self._client().chat.completions.create(
            **self._completion_arguments(prompt)
        )
        if not completion.choices:
            raise RuntimeError("LLM returned no completion choices")
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty completion content")
        return content.strip()

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield final-answer text fragments as the model produces them."""
        chunks = self._client().chat.completions.create(
            **self._completion_arguments(prompt),
            stream=True,
        )
        for chunk in chunks:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
