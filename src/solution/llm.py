from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel

from src.solution.artifacts import append_jsonl
from src.solution.constants import LLM_CALLS_PATH

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalise_input_artifacts(input_artifacts: Iterable[str | Path]) -> list[str]:
    return [str(path) for path in input_artifacts]


def log_llm_call(
    *,
    stage: str,
    provider: str,
    model: str,
    prompt: str,
    input_artifacts: Iterable[str | Path],
    output_artifact: str | Path,
    log_path: str | Path = LLM_CALLS_PATH,
) -> None:
    append_jsonl(
        log_path,
        {
            "stage": stage,
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_hash": hash_prompt(prompt),
            "input_artifacts": normalise_input_artifacts(input_artifacts),
            "output_artifact": str(output_artifact),
        },
    )


class StructuredLLM(Generic[SchemaT]):
    def __init__(
        self,
        *,
        schema: type[SchemaT],
        model_name: str | None = None,
        provider: str = "openrouter",
        log_path: str | Path = LLM_CALLS_PATH,
        structured_model=None,
    ) -> None:
        load_dotenv()
        self.schema = schema
        self.provider = provider
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4-5")
        self.log_path = Path(log_path)
        self._structured_model = structured_model

    def invoke(
        self,
        *,
        stage: str,
        prompt: str | ChatPromptTemplate,
        input_artifacts: Iterable[str | Path],
        output_artifact: str | Path,
        **prompt_kwargs,
    ) -> SchemaT:
        prompt_text = self._prompt_text(prompt, prompt_kwargs)
        result = self._model().invoke(self._messages(prompt, prompt_kwargs))

        if isinstance(result, self.schema):
            structured = result
        elif isinstance(result, dict) and "structured_response" in result:
            structured = self.schema.model_validate(result["structured_response"])
        else:
            structured = self.schema.model_validate(result)

        log_llm_call(
            stage=stage,
            provider=self.provider,
            model=self.model_name,
            prompt=prompt_text,
            input_artifacts=input_artifacts,
            output_artifact=output_artifact,
            log_path=self.log_path,
        )
        return structured

    def _model(self):
        if self._structured_model is None:
            self._structured_model = create_agent(
                model=self._chat_model(),
                tools=[],
                response_format=self.schema,
            )
        return self._structured_model

    def _chat_model(self):
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise RuntimeError("Install langchain-openrouter to use OpenRouter chat models") from exc
        return ChatOpenRouter(model=self.model_name, temperature=0)

    def _messages(self, prompt: str | ChatPromptTemplate, prompt_kwargs: dict) -> object:
        if isinstance(prompt, str):
            return {"messages": [HumanMessage(content=prompt)]}
        return {"messages": prompt.format_messages(**prompt_kwargs)}

    def _prompt_text(self, prompt: str | ChatPromptTemplate, prompt_kwargs: dict) -> str:
        if isinstance(prompt, str):
            return prompt
        messages = prompt.format_messages(**prompt_kwargs)
        return "\n".join(getattr(message, "content", str(message)) for message in messages)


def make_prompt(text: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(text)])
