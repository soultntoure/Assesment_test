from __future__ import annotations

import os
from typing import Generic, TypeVar

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseAgent(Generic[SchemaT]):
    def __init__(self, schema: type[SchemaT], *, model: str | None = None) -> None:
        load_dotenv()
        self.schema = schema
        self.model_name = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4-5")
        self._structured_model = None

    def invoke(self, prompt: str | ChatPromptTemplate, **kwargs) -> SchemaT:
        if isinstance(prompt, str):
            template = ChatPromptTemplate.from_messages(
                [HumanMessagePromptTemplate.from_template("{input}")]
            )
            messages = template.format_messages(input=prompt)
        else:
            messages = prompt.format_messages(**kwargs)
        result = self._agent().invoke({"messages": messages})
        structured = result.get("structured_response")
        if structured is None:
            raise RuntimeError(f"{self.schema.__name__} structured response was not returned")
        return self.schema.model_validate(structured)

    def _agent(self):
        if self._structured_model is None:
            self._structured_model = create_agent(model=self._chat_model(), tools=[], response_format=self.schema)
        return self._structured_model

    def _chat_model(self):
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise RuntimeError("Install langchain-openrouter to use OpenRouter chat models") from exc
        return ChatOpenRouter(model=self.model_name, temperature=0)