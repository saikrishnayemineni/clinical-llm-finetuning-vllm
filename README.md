<div align="center">

# 🧠 Domain-Specific Clinical LLM Fine-Tuning & vLLM Serving

[![FastAPI](https://img.shields.io/badge/FastAPI-OpenAI_Compatible-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![vLLM](https://img.shields.io/badge/vLLM-PagedAttention_2.0-792EE5?style=for-the-badge&logo=vllm&logoColor=white)](https://vllm.ai/)
[![PEFT DoRA](https://img.shields.io/badge/PEFT-DoRA_&_QLoRA-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

<p align="center">
  <b>Domain-Specific LLM Fine-Tuning and High-Throughput Serving Engine featuring 4-bit QLoRA + DoRA Weight Decomposition, Direct Preference Optimization (DPO), vLLM PagedAttention, and an OpenAI-compatible REST API.</b>
</p>

[✨ Live Model Battle Arena](http://localhost:8501) • [📚 OpenAI API (Swagger)](http://localhost:8000/docs) • [💼 Author Profile](https://github.com/saikrishnayemineni)

</div>

---

## 📌 Architecture & Mathematical Pipeline

```mermaid
graph TD
    subgraph 01. Preprocessing & Alignment Datasets
        RawEHR[Clinical Dialogues & SOAP Records] --> SFT_Data[Supervised Fine-Tuning - SFT Dataset]
        PhysicianReviews[Physician Preference Pairs - Chosen vs. Rejected] --> DPO_Data[DPO Alignment Dataset]
    end
    
    subgraph 02. QLoRA + DoRA Fine-Tuning Engine
        BaseModel[Base LLM Weights - 8B Parameters] --> NF4_Quant[4-Bit NormalFloat4 + Double Quantization]
        NF4_Quant --> DoRA_Layers[Weight-Decomposed Low-Rank Adapters - DoRA]
        SFT_Data --> SFT_Trainer[Stage 1: SFT Instruction Tuning]
        DoRA_Layers --> SFT_Trainer
        SFT_Trainer --> DPO_Trainer[Stage 2: Direct Preference Optimization - DPO]
        DPO_Data --> DPO_Trainer
        DPO_Trainer --> MergedAdapter[Trained Clinical DoRA/QLoRA Weights]
    end
    
    subgraph 03. High-Throughput vLLM PagedAttention Serving
        MergedAdapter --> PagedKVCache[PagedAttention 2.0 - Virtual Memory Block Manager]
        PagedKVCache --> ContinuousBatching[Continuous Batching & Chunked Prefill Scheduler]
        ContinuousBatching --> OpenAIServer[OpenAI-Compatible SSE API - /v1/chat/completions]
    end
    
    OpenAIServer --> ArenaUI([04. Streamlit 3-Model Clinical Battle Arena])
```

---

## 🔬 Mathematical Formulations

### 1. Weight-Decomposed Low-Rank Adaptation (DoRA):
$$W = m rac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c} = m rac{W_0 + rac{lpha}{r} B A}{\|W_0 + rac{lpha}{r} B A\|_c}$$

### 2. Direct Preference Optimization (DPO):
$$\mathcal{L}_{	ext{DPO}}(	heta; \pi_{	ext{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( eta \log rac{\pi_	heta(y_w \mid x)}{\pi_{	ext{ref}}(y_w \mid x)} - eta \log rac{\pi_	heta(y_l \mid x)}{\pi_{	ext{ref}}(y_l \mid x)} ight) ight]$$

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/saikrishnayemineni/clinical-llm-finetuning-vllm.git
cd clinical-llm-finetuning-vllm
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run 3-Model Arena Benchmark
```bash
python evaluate_finetuning.py
```

### 3. Launch Interactive Model Arena UI
```bash
streamlit run src/ui/app.py
```
Open: `http://localhost:8501`

### 4. Launch OpenAI-Compatible REST API
```bash
uvicorn src.api.main:app --reload --port 8000
```
Open: `http://localhost:8000/docs` (Use `/v1/chat/completions` like OpenAI SDK)

---

## 👨‍💻 Author

**Sai Krishna Yemineni** — *Production AI/ML Engineer*  
- Portfolio: [sai-krishna-portfolio-drab.vercel.app](https://sai-krishna-portfolio-drab.vercel.app)  
- LinkedIn: [linkedin.com/in/sai-krishna-yemineni](https://www.linkedin.com/in/sai-krishna-yemineni)  
- GitHub: [@saikrishnayemineni](https://github.com/saikrishnayemineni)
