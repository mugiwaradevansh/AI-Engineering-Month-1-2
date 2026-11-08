# A begineer - friendly terminal chat bot using Gemini , HTTPX, and Pydantic.

from __future__ import annotations

import asyncio
import json
import os
from types import TracebackType
from typing import Any,Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel,Field,ValidationError

# Gemini accepts "user" for user messages and "model" for its own replies.

MessageRole = Literal["user","model"]
class TextPart(BaseModel):
    # One text section inside a Gemini message
    text : str= Field(min_length=1)

class ChatMessage(BaseModel):
    # A validated chat message in Gemini rest API 
    role: MessageRole
    parts: list[TextPart]=Field(min_length=1)

class AsyncLLMClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key:str,
        model:str= "gemini-3.8-flash",
        timeout : float=60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("The Gemini API key is empty.")
        if not model.strip():
            raise ValueError("The Gemini model name is empty.")

        self.model = model
        self.endpoint = f"{self.BASE_URL}/{self.model}:streamGenerateContent"

        self._http = httpx.AsyncClient(
            headers={"x-goog-api-key":api_key},
            timeout = httpx.Timeout(timeout,connect=10.0),
        )

        self.history: list[ChatMessage] = []

        async def __aenter__(self) -> AsyncLLMClient:
            return self

        async def __aexit__(
                self,
                exc_type: type[BaseException]| None,
                exc : BaseException|None,
                traceback:TracebackType | None,
        ) -> None:
            await self.close()

        async def close(self) ->None:
            await self._http.aclose()

        @staticmethod
        def _extract_text(event:dict[str,Any]) -> str:
            candidates = event.get("candidates",[])
            if not candidates or not isinstance(candidates[0],dict):
                return ""
            content = candidates[0].get("content",{})

            if not isinstance(content,dict):
                return ""

            parts = content.get("parts",[])
            if not isinstance(parts,list):
                return ""

            return "".join(
                part["text"]
                for part in parts
                if isinstance(part,dict) and isinstance(part.get("text"),str)
            )

        