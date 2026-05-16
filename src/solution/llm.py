from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, TypeVar

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel

from src.solution.artifacts import append_jsonl
from src.solution.base_agent import BaseAgent
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


class StructuredLLM(BaseAgent[SchemaT]):
    def __init__(
        self,
        *,
        schema: type[SchemaT],
        model_name: str | None = None,
        provider: str = "gemini",
        log_path: str | Path = LLM_CALLS_PATH,
        structured_model=None,
    ) -> None:
        super().__init__(schema, model=model_name)
        self.provider = provider
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
        result = self._agent().invoke(self._messages(prompt, prompt_kwargs))

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
