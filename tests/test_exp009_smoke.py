from tac_sie.config import TACSIEConfig
from tac_sie.experiments import run_exp009


def test_exp009_smoke_reports_transfer_metrics():
    cfg = TACSIEConfig(device="cpu", d_hidden=64, n_memory_slots=4)
    metrics = run_exp009(cfg=cfg, train_steps=80, executor_epochs=250, seed=11)

    expected = {
        "carry_accuracy",
        "reset_accuracy",
        "shuffle_accuracy",
        "oracle_k_accuracy",
        "retrieved_k_accuracy",
        "known_rule_accuracy",
        "new_rule_accuracy",
        "same_query_counterfactual_accuracy",
        "avg_key_cosine",
        "correct_slot_attention",
        "correct_slot_margin",
        "read_attention_entropy",
        "offset_retrieval_accuracy",
    }
    assert expected <= metrics.keys()
    assert 0.0 <= metrics["carry_accuracy"] <= 1.0
    assert metrics["oracle_k_accuracy"] >= 0.95

