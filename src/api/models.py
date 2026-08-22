from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime

class ChatMessage(BaseModel):
    role: str  # 'system', 'user', 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="clinical-llama3-dora-8b")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(int(datetime.datetime.now(datetime.timezone.utc).timestamp())))
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]
