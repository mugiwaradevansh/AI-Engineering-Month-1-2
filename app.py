from fastapi import FastAPI

from schemas import ChatRequest,ChatResponse

app = FastAPI(
    title="My first AI chat API",
    description="A begineer friendly chat API built with FastAPI and Pydantic.",
    version="1.0.0",
)

@app.post("/chat",response_model=ChatResponse)
def chat(request:ChatRequest) -> ChatResponse:
    if request.message:
        last_message = request.message[-1]

        reply =(
            f"{last_message.content}"
        )
    else:
        reply = "Mock response : Your message list is empty."

    return ChatResponse(
        reply=reply,
        model_used=request.model,
        token_used=0,
    )