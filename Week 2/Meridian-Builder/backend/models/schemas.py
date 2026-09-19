# models/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str

class Finding(BaseModel):
    kind: Literal["phone", "money", "time"]
    value: str

class ChatResponse(BaseModel):
    reply: str
    ungrounded: list[Finding] # empty means fully grounded
    latency_ms: int
    tokens_in: int
    tokens_out: int