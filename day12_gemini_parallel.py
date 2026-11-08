
import asyncio
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv


# Load the API key from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_KEY")

MODEL = "gemini-3.8-flash"

URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:generateContent"
)

REQUEST_TIMEOUT = 5.0


# Three different prompts for three real API requests
PROMPTS = [
    "Explain asynchronous programming in Python in 2 simple sentences.",
    "Explain Pydantic and why FastAPI uses it in 2 simple sentences.",
    "Explain the difference between an HTTP error and a timeout in 2 simple sentences.",
]


async def generate_content(
    client: httpx.AsyncClient,
    prompt: str,
    request_number: int,
) -> str:
    """Send one asynchronous request to the Gemini API."""

    print(f"[Request {request_number}] Started")

    headers = {
        "x-goog-api-key": API_KEY or "",
        "Content-Type": "application/json",
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 120
        },
    }

    response = await client.post(
        URL,
        headers=headers,
        json=body,
    )

    response.raise_for_status()

    data: dict[str, Any] = response.json()

    generated_text = (
        data["candidates"][0]
        ["content"]["parts"][0]["text"]
    )

    print(f"[Request {request_number}] Completed")

    return generated_text


async def generate_with_timeout(
    client: httpx.AsyncClient,
    prompt: str,
    request_number: int,
) -> str | None:
    """Run one Gemini request with its own 5-second timeout."""

    try:
        result = await asyncio.wait_for(
            generate_content(
                client,
                prompt,
                request_number,
            ),
            timeout=REQUEST_TIMEOUT,
        )

        return result

    except TimeoutError:
        print(
            f"[Request {request_number}] "
            f"Timed out after {REQUEST_TIMEOUT} seconds."
        )

    except httpx.TimeoutException as error:
        print(
            f"[Request {request_number}] "
            f"HTTP client timeout: {error}"
        )

    except httpx.HTTPStatusError as error:
        print(
            f"[Request {request_number}] "
            f"HTTP {error.response.status_code}: "
            f"{error.response.text[:500]}"
        )

    except httpx.RequestError as error:
        print(
            f"[Request {request_number}] "
            f"Network error: {error}"
        )

    except (KeyError, IndexError, TypeError, ValueError) as error:
        print(
            f"[Request {request_number}] "
            f"Response parsing error: {error}"
        )

    return None


async def run_parallel_requests() -> None:
    """Execute three Gemini requests concurrently."""

    if not API_KEY:
        print("Error: GEMINI_KEY is missing from .env")
        return

    print("\nStarting three Gemini API requests...\n")

    start_time = time.perf_counter()

    # Keep one HTTP client open for all three requests.
    async with httpx.AsyncClient(timeout=20.0) as client:

        tasks = [
            generate_with_timeout(
                client,
                prompt,
                request_number,
            )
            for request_number, prompt in enumerate(
                PROMPTS,
                start=1,
            )
        ]

        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    print("\n========== GEMINI RESULTS ==========")

    for request_number, result in enumerate(results, start=1):
        print(f"\n--- Request {request_number} ---")

        if result is None:
            print("No response received.")
        else:
            print(result)

    successful_requests = sum(
        result is not None for result in results
    )

    print("\n========== SUMMARY ==========")
    print(f"Total requests: {len(PROMPTS)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed or timed out: {len(PROMPTS) - successful_requests}")
    print(f"Total execution time: {total_time:.2f} seconds")


def main() -> None:
    """Start the asynchronous application."""

    asyncio.run(run_parallel_requests())


if __name__ == "__main__":
    main()
