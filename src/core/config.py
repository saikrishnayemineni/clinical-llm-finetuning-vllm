import os

try:
    from pydantic_settings import BaseSettings
    class FineTuningSettings(BaseSettings):
        PROJECT_NAME: str = "Domain-Specific Clinical LLM Fine-Tuning & vLLM Serving"
        VERSION: str = "1.0.0"
        BASE_MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"
        LORA_RANK: int = 16
        LORA_ALPHA: int = 32
        LORA_DROPOUT: float = 0.05
        USE_DORA: bool = True
        QUANTIZATION_BITS: int = 4  # 4-bit NF4
        DPO_BETA: float = 0.1
        VLLM_BLOCK_SIZE: int = 16
        MAX_MODEL_LEN: int = 4096
        class Config:
            case_sensitive = True
    settings = FineTuningSettings()
except ImportError:
    class FineTuningSettings:
        PROJECT_NAME: str = "Domain-Specific Clinical LLM Fine-Tuning & vLLM Serving"
        VERSION: str = "1.0.0"
        BASE_MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"
        LORA_RANK: int = 16
        LORA_ALPHA: int = 32
        LORA_DROPOUT: float = 0.05
        USE_DORA: bool = True
        QUANTIZATION_BITS: int = 4
        DPO_BETA: float = 0.1
        VLLM_BLOCK_SIZE: int = 16
        MAX_MODEL_LEN: int = 4096
    settings = FineTuningSettings()
