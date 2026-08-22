import pytest
from src.finetuning.dpo_trainer import DirectPreferenceOptimizationTrainer

def test_dpo_loss_calculation():
    trainer = DirectPreferenceOptimizationTrainer(beta=0.1)
    loss = trainer.compute_dpo_loss(-0.4, -1.8, -0.6, -1.2)
    assert loss > 0.0
    assert loss < 2.0
