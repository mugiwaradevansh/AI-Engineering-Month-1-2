
import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_KEY")

URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash:generateContent"
)

TIMEOUT = 30.0

headers = {
    "x-goog-api-key": API_KEY or "",
    "Content-Type": "application/json",
}

body = {
    "contents": [
        {
            "parts": [
                {
                    "text": (
                        "Explain what an AI Engineer does "
                        "in 5 simple sentences."
                    )
                }
            ]
        }
    ]
}


async def call_gemini() -> None:
    """Call Gemini using an asynchronous HTTP request."""

    if not API_KEY:
        print("Error: GEMINI_KEY is missing from .env")
        return

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:

            response = await client.post(
                URL,
                headers=headers,
                json=body,
            )

            print("HTTP Status:", response.status_code)

            response.raise_for_status()

            data: dict[str, Any] = response.json()

            generated_text = (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

            print("\nGemini response:")
            print(generated_text)

    except httpx.TimeoutException:
        print("Error: Gemini API request timed out.")

    except httpx.HTTPStatusError as error:
        print("HTTP Error:", error.response.status_code)
        print("Error details:", error.response.text)

    except httpx.RequestError as error:
        print("Network error:", error)

    except (KeyError, IndexError, TypeError, ValueError) as error:
        print("Error while parsing Gemini response:", error)


async def main() -> None:
    """Run the asynchronous Gemini API call."""
    await call_gemini()


if __name__ == "__main__":
    asyncio.run(main())
