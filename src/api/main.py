import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.serving.vllm_engine import vllm_serving_engine
from src.api.models import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="OpenAI-Compatible High-Speed vLLM Serving Server for Fine-Tuned Clinical LLMs.",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "engine": "vLLM with PagedAttention 2.0",
        "loaded_models": ["base_llama3_8b", "sft_clinical_8b", "clinical-llama3-dora-8b"]
    }

@app.get("/v1/models", tags=["OpenAI Models"])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "clinical-llama3-dora-8b", "object": "model", "owned_by": "saikrishnayemineni"},
            {"id": "sft_clinical_8b", "object": "model", "owned_by": "saikrishnayemineni"},
            {"id": "base_llama3_8b", "object": "model", "owned_by": "meta"}
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["OpenAI Chat Completions"])
async def chat_completions(request: ChatCompletionRequest):
    last_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    
    # Map model name
    m_type = "dpo_dora_clinical"
    if "sft" in request.model.lower():
        m_type = "sft_clinical"
    elif "base" in request.model.lower():
        m_type = "base_8b"

    res = vllm_serving_engine.generate(last_user_msg, model_type=m_type)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=res["generated_text"]),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(last_user_msg.split()),
            "completion_tokens": res["tokens_generated"],
            "total_tokens": len(last_user_msg.split()) + res["tokens_generated"]
        }
    )
