import unittest

from experiments.benchmark_identity_attention_selectivity import (
    aggregate_identity_attention_runs,
    identity_attention_decision,
    variant_config_overrides,
)


class IdentityAttentionSelectivityTests(unittest.TestCase):
    def test_variant_overrides_include_current_and_selective_modes(self):
        variants = variant_config_overrides(attention_window_size=4)

        self.assertEqual(variants["identity_first"]["identity_attention_type"], "identity_first")
        self.assertEqual(variants["compressed_memory"]["identity_attention_type"], "compressed_memory")
        self.assertEqual(variants["coherence_sparse"]["identity_attention_type"], "coherence_sparse")
        self.assertEqual(
            variants["coherence_sparse_local"]["identity_attention_type"],
            "coherence_sparse",
        )
        self.assertEqual(variants["coherence_sparse_local"]["attention_window_size"], 4)

    def test_decision_promotes_only_when_quality_improves_without_speed_loss(self):
        runs = [
            _run("identity_first", "multi_hop", carry=0.40, utility=0.20, tps=100.0),
            _run("identity_first", "single_key", carry=0.80, utility=0.30, tps=100.0),
            _run("coherence_sparse_local", "multi_hop", carry=0.45, utility=0.25, tps=104.0),
            _run("coherence_sparse_local", "single_key", carry=0.82, utility=0.31, tps=104.0),
            _run("compressed_memory", "multi_hop", carry=0.47, utility=0.24, tps=92.0),
            _run("compressed_memory", "single_key", carry=0.83, utility=0.31, tps=92.0),
        ]

        aggregate = aggregate_identity_attention_runs(
            runs,
            baseline_variant="identity_first",
            min_quality_gain=0.02,
            min_speed_ratio=0.98,
        )

        self.assertEqual(aggregate["decision"]["status"], "promote_selective_identity_attention")
        self.assertEqual(aggregate["decision"]["promoted_variant"], "coherence_sparse_local")
        self.assertIn("compressed_memory", aggregate["rejected_variants"])

    def test_decision_rejects_when_quality_gain_costs_speed(self):
        decision = identity_attention_decision(
            baseline={"mean_quality": 0.60, "mean_eval_tps": 100.0},
            candidate={"mean_quality": 0.65, "mean_eval_tps": 90.0},
            variant="compressed_memory",
            min_quality_gain=0.02,
            min_speed_ratio=0.98,
        )

        self.assertEqual(decision["status"], "reject")
        self.assertFalse(decision["checks"]["speed_not_hurt"])


def _run(variant, task, *, carry, utility, tps):
    return {
        "variant": variant,
        "task": task,
        "seed": 11,
        "result": {
            "tac": {
                "chunked_probe": {
                    "carry": {
                        "value_accuracy": carry,
                        "tokens_per_second": tps,
                    }
                },
            },
            "decision": {
                "value_accuracy_delta": utility,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
