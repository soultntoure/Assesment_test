from __future__ import annotations

import os
from typing import Generic, TypeVar

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel

from src.solution.constants import DEFAULT_GEMINI_MODEL

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseAgent(Generic[SchemaT]):
    def __init__(self, schema: type[SchemaT], *, model: str | None = None) -> None:
        load_dotenv()
        self.schema = schema
        self.model_name = model or DEFAULT_GEMINI_MODEL
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
            self._structured_model = create_agent(
                model=self._chat_model(),
                tools=[],
                response_format=self.schema,
            )
        return self._structured_model

    def _chat_model(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key is None:
            raise RuntimeError("Set GEMINI_API_KEY to the API key from Google AI Studio")

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
            google_api_key=gemini_api_key,
        )
