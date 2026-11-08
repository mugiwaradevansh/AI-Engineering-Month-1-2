
from typing import List

from pydantic import BaseModel, ValidationError


class ChatMessage(BaseModel):
    """Represents one message in a chat conversation."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Represents a complete chat API request."""

    messages: List[ChatMessage]
    model: str = "gpt-4o"
    temperature: float = 0.7


def main() -> None:
    """Create and validate chat request models."""

    # 1. Create a valid chat request
    valid_request = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Explain what an API is.",
            )
        ]
    )

    print("=== VALID REQUEST ===")
    print(valid_request)

    print("\n=== REQUEST AS DICTIONARY ===")
    print(valid_request.model_dump())

    # 2. Intentionally pass incorrect data
    invalid_data = {
        "messages": [
            {
                "role": 123,
                "content": ["Hello", "World"],
            }
        ],
        "model": 123,
        "temperature": "very high",
    }

    print("\n=== INVALID REQUEST ===")

    try:
        invalid_request = ChatRequest(**invalid_data)
        print(invalid_request)

    except ValidationError as error:
        print(error)


if __name__ == "__main__":
    main()
