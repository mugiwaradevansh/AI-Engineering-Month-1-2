from typing import List

from pydantic import BaseModel,Field

class ChatMessage(BaseModel):
    role:str
    content:str

class ChatRequest(BaseModel):
    message:List[ChatMessage]
    model:str = "gpt-4o"
    temperature:float=Field(
        default = 0.7,
        ge = 0.0,
        le = 2.0,
    )

class ChatResponse(BaseModel):
    reply:str
    model_used:str
    token_used:int