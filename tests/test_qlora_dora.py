import pytest
from src.finetuning.qlora_dora import DoRAWeightDecomposer

def test_dora_forward_pass():
    # Realistic 4096 hidden dimension for Llama-3-8B
    dora = DoRAWeightDecomposer(in_features=4096, out_features=4096, rank=16, alpha=32)
    x = [0.5] * 64
    out = dora.forward(x)
    assert len(out) == 64
    summary = dora.get_parameter_summary()
    assert summary["trainable_percentage"] < 3.0  # ~0.80% parameter efficiency
