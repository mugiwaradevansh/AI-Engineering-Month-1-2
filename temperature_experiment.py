
"""
Experiment: Temperature vs Maximum Output Tokens

Experiments:
1. Temperature 0.0  -> 5 responses
2. Temperature 0.5  -> 5 responses
3. Temperature 1.0  -> 5 responses
4. Temperature 1.5  -> 5 responses
5. Maximum output tokens = 20
6. Maximum output tokens = 500

Total: 22 API requests.

Required .env variables:
GEMINI_KEY=your_actual_api_key
GEMINI_TEST_MODEL=gemini-2.5-flash
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv


# --------------------------------------------------
# 1. EXPERIMENT CONFIGURATION
# --------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"

# We will send this EXACT same prompt in every request.
PROMPT = (
    "Write a short, imaginative story about a tiny robot "
    "that discovers a mysterious door beneath an old banyan tree. "
    "Give the story a clear beginning, middle, and ending. "
    "Aim for about 100 words."
)

# Four temperature values.
TEMPERATURES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)

# Repeat each temperature experiment five times.
REPETITIONS: int = 5


# --------------------------------------------------
# 2. EXTRACT TEXT FROM GEMINI RESPONSE
# --------------------------------------------------

def extract_response_text(data: dict[str, Any]) -> str:
    """Extract generated text from a Gemini API response."""

    candidates = data.get("candidates", [])

    if not candidates:
        prompt_feedback = data.get("promptFeedback", {})
        return f"[No candidate returned: {prompt_feedback}]"

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    text_parts: list[str] = []

    for part in parts:
        if isinstance(part, dict) and isinstance(
            part.get("text"), str
        ):
            text_parts.append(part["text"])

    if not text_parts:
        finish_reason = candidates[0].get(
            "finishReason", "unknown"
        )
        return f"[No text returned. Finish reason: {finish_reason}]"

    return "".join(text_parts)


# --------------------------------------------------
# 3. SEND ONE REQUEST TO GEMINI
# --------------------------------------------------

async def generate_response(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """Send one asynchronous request and return Gemini's response."""

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    # Prepare the JSON body sent to Gemini.
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,

            # Gemini uses maxOutputTokens,
            # not OpenAI's max_tokens parameter.
            "maxOutputTokens": max_output_tokens,
        },
    }

    # Send the HTTP POST request asynchronously.
    response = await client.post(
        url,
        json=payload
    )

    # Raise an exception if HTTP returned an error.
    response.raise_for_status()

    # Convert JSON response into a Python dictionary.
    data = response.json()

    # Extract and return only generated text.
    return extract_response_text(data)


# --------------------------------------------------
# 4. EXPERIMENT ONE: TEMPERATURE
# --------------------------------------------------

async def run_temperature_experiment(
    client: httpx.AsyncClient,
    model: str,
) -> None:
    """Run the same prompt five times at each temperature."""

    print("\n" + "=" * 72)
    print("EXPERIMENT 1: TEMPERATURE")
    print("=" * 72)

    print(f"Model: {model}")
    print(f"Prompt: {PROMPT}\n")

    # Outer loop: change temperature.
    for temperature in TEMPERATURES:

        print("\n" + "-" * 72)
        print(
            f"TEMPERATURE = {temperature} "
            f"| {REPETITIONS} repetitions"
        )
        print("-" * 72)

        # Inner loop: repeat five times.
        for repetition in range(1, REPETITIONS + 1):

            print(
                f"\n[Temperature {temperature} "
                f"| Run {repetition}]"
            )

            try:
                answer = await generate_response(
                    client=client,
                    model=model,
                    prompt=PROMPT,
                    temperature=temperature,
                    max_output_tokens=180,
                )

                print(answer)

            except httpx.HTTPStatusError as error:
                print(
                    f"[HTTP error: "
                    f"{error.response.status_code}]"
                )
                print(error.response.text[:1200])

            except httpx.RequestError as error:
                print(f"[Network error: {error}]")


# --------------------------------------------------
# 5. EXPERIMENT TWO: MAXIMUM OUTPUT TOKENS
# --------------------------------------------------

async def run_output_limit_experiment(
    client: httpx.AsyncClient,
    model: str,
) -> None:
    """Compare output limits of 20 and 500 tokens."""

    print("\n" + "=" * 72)
    print("EXPERIMENT 2: MAXIMUM OUTPUT TOKENS")
    print("=" * 72)

    print(f"Same prompt: {PROMPT}")
    print("Temperature is fixed at 1.0.\n")

    # Test two different maximum output limits.
    for max_output_tokens in (20, 500):

        print("\n" + "-" * 72)
        print(
            f"MAXIMUM OUTPUT TOKENS = {max_output_tokens}"
        )
        print("-" * 72)

        try:
            answer = await generate_response(
                client=client,
                model=model,
                prompt=PROMPT,
                temperature=1.0,
                max_output_tokens=max_output_tokens,
            )

            print(answer)

        except httpx.HTTPStatusError as error:
            print(
                f"[HTTP error: "
                f"{error.response.status_code}]"
            )
            print(error.response.text[:1200])

        except httpx.RequestError as error:
            print(f"[Network error: {error}]")


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

async def main() -> None:
    """Load API credentials and execute both experiments."""

    # Load environment variables from .env.
    load_dotenv()

    api_key = (
        os.getenv("GEMINI_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    model = os.getenv(
        "GEMINI_TEST_MODEL",
        DEFAULT_MODEL
    ).strip()

    if not api_key:
        print(
            "Missing API key. Add GEMINI_KEY "
            "to your .env file."
        )
        return

    print("\nGemini Parameter Experiment")
    print("Total requests: Up to 22")
    print(f"Selected model: {model}")

    try:
        # Create ONE HTTP client and reuse it.
        async with httpx.AsyncClient(
            headers={
                "x-goog-api-key": api_key
            },
            timeout=httpx.Timeout(
                90.0,
                connect=15.0
            ),
        ) as client:

            # Run temperature experiment.
            await run_temperature_experiment(
                client,
                model
            )

            # Run maximum output token experiment.
            await run_output_limit_experiment(
                client,
                model
            )

    except httpx.HTTPStatusError as error:
        print(
            f"\nAPI error: "
            f"{error.response.status_code}"
        )
        print(error.response.text[:1500])

    except httpx.RequestError as error:
        print(f"\nConnection error: {error}")


# --------------------------------------------------
# 7. OBSERVATIONS — UPDATE AFTER RUNNING
# --------------------------------------------------

# Expected observations (verify them using actual responses):
#
# 1. Temperature 0.0:
#    Usually produces more consistent responses.
#    Identical responses are NOT guaranteed.
#
# 2. Temperature 0.5:
#    May introduce some variation while staying focused.
#
# 3. Temperature 1.0:
#    May produce more varied wording and ideas.
#
# 4. Temperature 1.5:
#    May produce more unpredictable responses.
#    Higher temperature does NOT automatically mean better creativity.
#
# 5. maxOutputTokens = 20:
#    Creates a small output ceiling.
#    The story may finish abruptly.
#
# 6. maxOutputTokens = 500:
#    Allows a longer response, but does NOT force Gemini
#    to generate 500 tokens.
#
# 7. Compare the five actual responses at every temperature.
#    Record what you observe instead of assuming the results.


# --------------------------------------------------
# 8. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nExperiment stopped by user.")

