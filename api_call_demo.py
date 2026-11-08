
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

TIMEOUT = 30


def call_gemini(prompt: str) -> None:
    """Send a prompt to Gemini using a direct HTTP POST request."""

    api_key = os.getenv("GEMINI_KEY")

    if not api_key:
        print("GEMINI_KEY is missing from your .env file.")
        return

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.8-flash:generateContent"
    )

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        data: dict[str, Any] = response.json()

        generated_text = (
            data["candidates"][0]["content"]["parts"][0]["text"]
        )

        print("\n=== GEMINI RESPONSE ===")
        print(generated_text)

    except requests.RequestException as error:
        print(f"Gemini request failed: {error}")

    except (KeyError, IndexError, TypeError) as error:
        print(f"Unexpected Gemini response structure: {error}")


def call_openai(prompt: str) -> None:
    """Send a prompt to OpenAI using a direct HTTP POST request."""

    api_key = os.getenv("OPENAI_KEY")

    if not api_key:
        print("\nOPENAI_KEY not found. Skipping OpenAI.")
        return

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        data: dict[str, Any] = response.json()

        generated_text = (
            data["choices"][0]["message"]["content"]
        )

        print("\n=== OPENAI RESPONSE ===")
        print(generated_text)

    except requests.RequestException as error:
        print(f"OpenAI request failed: {error}")

    except (KeyError, IndexError, TypeError) as error:
        print(f"Unexpected OpenAI response structure: {error}")


def main() -> None:
    """Run the Gemini and optional OpenAI API calls."""

    prompt = (
        "What is Universe and how universe come in existence ? If universe is became by Big bang so for big bang that atoms are came from where is in simple English. "
        "Give one real-world example."
    )

    call_gemini(prompt)
    call_openai(prompt)


if __name__ == "__main__":
    main()
