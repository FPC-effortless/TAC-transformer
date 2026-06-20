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
        "known_rule_shuffle_accuracy",
        "new_rule_shuffle_accuracy",
        "known_rule_reset_accuracy",
        "new_rule_reset_accuracy",
        "same_query_counterfactual_accuracy",
        "counterfactual_drop",
        "known_rule_counterfactual_drop",
        "new_rule_counterfactual_drop",
        "avg_key_cosine",
        "correct_slot_attention",
        "correct_slot_margin",
        "read_attention_entropy",
        "offset_retrieval_accuracy",
    }
    assert expected <= metrics.keys()
    assert 0.0 <= metrics["carry_accuracy"] <= 1.0
    assert metrics["oracle_k_accuracy"] >= 0.95


def test_exp009_counterfactual_metric_is_not_alias_of_new_rule_success():
    cfg = TACSIEConfig(device="cpu", d_hidden=64, n_memory_slots=4)
    metrics = run_exp009(cfg=cfg, train_steps=80, executor_epochs=250, seed=12)

    # same_query_counterfactual_accuracy is the new-rule wrong-binding control.
    # It must not be copied from new_rule_accuracy or from the known-rule shuffle
    # metric, because that would overstate counterfactual evidence in the audit.
    assert metrics["same_query_counterfactual_accuracy"] == metrics["new_rule_shuffle_accuracy"]
    assert metrics["new_rule_counterfactual_drop"] == (
        metrics["new_rule_accuracy"] - metrics["same_query_counterfactual_accuracy"]
    )
    assert metrics["known_rule_counterfactual_drop"] == (
        metrics["known_rule_accuracy"] - metrics["known_rule_shuffle_accuracy"]
    )
