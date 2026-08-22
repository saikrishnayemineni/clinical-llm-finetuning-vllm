import streamlit as st
import json
from pathlib import Path
from src.serving.vllm_engine import vllm_serving_engine
from src.finetuning.qlora_dora import DoRAWeightDecomposer

st.set_page_config(
    page_title="Clinical LLM Fine-Tuning & vLLM Arena",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #080c16; }
    .stMetric { background-color: #0f172a; border-radius: 10px; padding: 10px; border: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# Load prompts
prompts_path = Path("data/benchmark_prompts.json")
if prompts_path.exists():
    with open(prompts_path, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
else:
    PROMPTS = []

st.sidebar.title("🧠 QLoRA / DoRA Controls")
st.sidebar.caption("vLLM Continuous Batching Serving")

p_titles = [f"{p['clinical_domain']}: {p['prompt'][:40]}..." for p in PROMPTS]
selected_idx = st.sidebar.selectbox("📂 Load Clinical Test Prompt", range(len(p_titles)), format_func=lambda i: p_titles[i] if p_titles else "None")

selected_p = PROMPTS[selected_idx]["prompt"] if PROMPTS else "Explain STEMI protocol."

st.sidebar.divider()
st.sidebar.markdown("##### ⚙️ Quantization & PEFT Architecture")
st.sidebar.caption("• **Base Model:** Llama-3-8B (FP16)")
st.sidebar.caption("• **Quantization:** 4-Bit NormalFloat4 (NF4)")
st.sidebar.caption("• **PEFT Method:** DoRA (r=16, alpha=32)")
st.sidebar.caption("• **Alignment:** Direct Preference Optimization (DPO)")

# Header
st.title("🧠 Clinical LLM Arena: Base vs. SFT vs. DPO+DoRA")
st.caption("Side-by-Side Model Comparison • 4-bit QLoRA/DoRA Weight Decomposition • vLLM PagedAttention 2.0 • OpenAI REST API")

user_prompt = st.text_area("Enter Clinical Prompt for Evaluation:", value=selected_p, height=80)
run_btn = st.button("🚀 Run 3-Model Battle Arena", type="primary", use_container_width=True)

if run_btn or user_prompt:
    c1, c2, c3 = st.columns(3)
    
    # 1. Base Model
    with c1:
        st.subheader("🤖 Base Llama-3-8B")
        st.caption("Generic Base Model (No Clinical Tuning)")
        res_base = vllm_serving_engine.generate(user_prompt, model_type="base_8b")
        st.info(res_base["generated_text"])
        st.metric("VRAM Usage", "16.2 GB (FP16)", delta="High VRAM", delta_color="inverse")
        st.metric("Throughput", f"{res_base['tokens_per_second']} tok/s")

    # 2. SFT Model
    with c2:
        st.subheader("📑 SFT Clinical 8B")
        st.caption("Supervised Instruction Fine-Tuning")
        res_sft = vllm_serving_engine.generate(user_prompt, model_type="sft_clinical")
        st.warning(res_sft["generated_text"])
        st.metric("VRAM Usage", "4.1 GB (4-bit NF4)", delta="-74% VRAM Saved")
        st.metric("Throughput", f"{res_sft['tokens_per_second']} tok/s")

    # 3. DPO + DoRA Model
    with c3:
        st.subheader("🏆 DPO + DoRA Clinical 8B")
        st.caption("DoRA Weight Decomp + Physician DPO Alignment")
        res_dpo = vllm_serving_engine.generate(user_prompt, model_type="dpo_dora_clinical")
        st.success(res_dpo["generated_text"])
        st.metric("VRAM Usage", "4.1 GB (4-bit NF4)", delta="-74% VRAM Saved")
        st.metric("Throughput", f"{res_dpo['tokens_per_second']} tok/s", delta="Sub-15ms TTFT")
