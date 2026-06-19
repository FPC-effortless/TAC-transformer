from tac_sie.experiments import run_exp009b


def test_exp009b_smoke_reports_leak_controls():
    result = run_exp009b(
        seeds=[0],
        n_memory_slots_values=[2],
        n_offsets_values=[2],
        train_steps=20,
        executor_epochs=80,
        batch_size=32,
        device="cpu",
    )

    summary = result["summary"]
    assert result["rows"]
    assert "wrong_offset_accuracy" in summary
    assert "wrong_rule_state_accuracy" in summary
    assert "swapped_state_accuracy" in summary
    assert "random_query_rule_accuracy" in summary
    assert 0.0 <= summary["carry_accuracy"] <= 1.0
