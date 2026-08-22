import pytest
from src.finetuning.qlora_dora import DoRAWeightDecomposer

def test_dora_forward_pass():
    dora = DoRAWeightDecomposer(in_features=64, out_features=64, rank=8, alpha=16)
    x = [0.5] * 64
    out = dora.forward(x)
    assert len(out) == 64
    summary = dora.get_parameter_summary()
    assert summary["trainable_percentage"] < 3.0  # < 3% trainable parameter efficiency
