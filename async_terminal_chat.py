
"""A beginner-friendly terminal chatbot using Gemini, HTTPX, and Pydantic."""

from __future__ import annotations

import asyncio
import json
import os
from types import TracebackType
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


# Gemini accepts "user" for user messages and "model" for its own replies.
MessageRole = Literal["user", "model"]


class TextPart(BaseModel):
    """One text section inside a Gemini message."""

    text: str = Field(min_length=1)


class ChatMessage(BaseModel):
    """A validated chat message in Gemini's REST API format."""

    role: MessageRole
    parts: list[TextPart] = Field(min_length=1)


class AsyncLLMClient:
    """Keep chat history and send messages to Gemini using async HTTPX."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 120.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("The Gemini API key is empty.")
        if not model.strip():
            raise ValueError("The Gemini model name is empty.")

        self.model = model
        self.endpoint = f"{self.BASE_URL}/{self.model}:streamGenerateContent"

        # This one HTTP client is reused for every message in the chat.
        self._http = httpx.AsyncClient(
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

        # This list stays in memory while this application is running.
        self.history: list[ChatMessage] = []

    async def __aenter__(self) -> AsyncLLMClient:
        """Allow this client to be used with 'async with'."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the HTTP client when the 'async with' block ends."""
        await self.close()

    async def close(self) -> None:
        """Release the resources used by the HTTP client."""
        await self._http.aclose()

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str:
        """Read the generated text from one Gemini streaming event."""
        candidates = event.get("candidates", [])
        if not candidates or not isinstance(candidates[0], dict):
            return ""

        content = candidates[0].get("content", {})
        if not isinstance(content, dict):
            return ""

        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return ""

        return "".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )

    async def send_message(self, prompt: str) -> str:
        """Send one user message, stream the reply, and save both turns."""
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Your message cannot be empty.")

        user_message = ChatMessage(
            role="user",
            parts=[TextPart(text=clean_prompt)],
        )

        # Create temporary history for this request.
        pending_history = [*self.history, user_message]

        payload: dict[str, Any] = {
            "contents": [message.model_dump() for message in pending_history]
        }

        received_chunks: list[str] = []

        async with self._http.stream(
            method="POST",
            url=self.endpoint,
            params={"alt": "sse"},
            json=payload,
        ) as response:

            if response.is_error:
                await response.aread()
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                json_text = line[len("data:"):].strip()

                if not json_text or json_text == "[DONE]":
                    continue

                try:
                    event: Any = json.loads(json_text)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Gemini sent an invalid streaming event: {json_text[:200]}"
                    ) from error

                if not isinstance(event, dict):
                    continue

                if "error" in event:
                    raise RuntimeError(f"Gemini streaming error: {event['error']}")

                chunk_text = self._extract_text(event)

                if chunk_text:
                    received_chunks.append(chunk_text)
                    print(chunk_text, end="", flush=True)

        print()

        full_reply = "".join(received_chunks)

        if not full_reply:
            raise RuntimeError(
                "Gemini finished without returning text. Check the model, "
                "API access, or whether the response was blocked."
            )

        # Save the exchange only after streaming succeeds.
        assistant_message = ChatMessage(
            role="model",
            parts=[TextPart(text=full_reply)],
        )

        self.history.extend([user_message, assistant_message])

        return full_reply


async def main() -> None:
    """Load settings and run the interactive terminal chat loop."""
    load_dotenv()

    api_key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    try:
        timeout = float(os.getenv("GEMINI_TIMEOUT", "120"))
    except ValueError as error:
        raise ValueError("GEMINI_TIMEOUT must be a number of seconds.") from error
    if timeout <= 0:
        raise ValueError("GEMINI_TIMEOUT must be greater than zero.")

    if not api_key:
        print("Missing API key. Add GEMINI_KEY=your_key to your .env file.")
        return

    print("\n====================================")
    print("       ASYNC TERMINAL AI CHAT")
    print("====================================")
    print(f"Model: {model}")
    print("Commands: /help, /reset, /exit")
    print("Your conversation history lasts until you exit this program.\n")

    try:
        async with AsyncLLMClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
        ) as llm:

            while True:
                try:
                    user_text = await asyncio.to_thread(input, "You: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                command = user_text.strip().lower()

                if command in {"/exit", "/quit"}:
                    print("Goodbye!")
                    break

                if command == "/help":
                    print("\nCommands:")
                    print("  /help  - show these commands")
                    print("  /reset - clear the conversation history")
                    print("  /exit  - quit the chat\n")
                    continue

                if command == "/reset":
                    llm.history.clear()
                    print("Conversation history cleared.\n")
                    continue

                if not user_text.strip():
                    continue

                print("Gemini: ", end="", flush=True)

                try:
                    await llm.send_message(user_text)

                except httpx.HTTPStatusError as error:
                    print(f"\n[Gemini API HTTP error: {error.response.status_code}]")
                    print(error.response.text)

                except httpx.TimeoutException:
                    print("\n[Timeout] Gemini took too long to respond. Try again.")

                except httpx.RequestError as error:
                    print(f"\n[Network error] {error}")

                except (RuntimeError, ValidationError, ValueError) as error:
                    print(f"\n[Chat error] {error}")

                except Exception as error:
                    print(f"\n[Unexpected error] {type(error).__name__}: {error}")

    except ValueError as error:
        print(f"Configuration error: {error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat stopped. Goodbye!")
