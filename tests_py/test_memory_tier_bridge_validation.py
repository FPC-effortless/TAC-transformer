import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import count_parameters, estimate_tac_parameter_count


def _ablate_semantic_procedural(states):
    ablated = []
    for state in states:
        ablated.append(
            replace(
                state,
                semantic_state=(
                    torch.zeros_like(state.semantic_state)
                    if state.semantic_state is not None
                    else None
                ),
                procedural_state=(
                    torch.zeros_like(state.procedural_state)
                    if state.procedural_state is not None
                    else None
                ),
            )
        )
    return ablated


class MemoryTierBridgeValidationTests(unittest.TestCase):
    def test_semantic_procedural_bridge_is_model_native_and_trainable(self):
        torch.manual_seed(13)
        config = TACConfig(
            vocab_size=48,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=4,
            routing_type="base_semantic",
            routing_top_k=2,
            memory_system_type="multi_timescale",
            memory_bridge_type="semantic_procedural_readout",
            memory_bridge_weight=1.5,
            identity_attention_type="none",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 8, 24, 4]]), collect_auxiliary=True)
        query = torch.tensor([[2, 8]])
        labels = torch.tensor([[-100, 24]])

        carried = model(
            query,
            identity_states=context.identity_states,
            labels=labels,
            collect_auxiliary=True,
        )
        ablated = model(
            query,
            identity_states=_ablate_semantic_procedural(context.identity_states),
            collect_auxiliary=True,
        )
        carried.loss.backward()

        self.assertIn("memory_bridge_update_norm", carried.aux.metrics)
        self.assertIn("memory_bridge_tier_entropy", carried.aux.metrics)
        self.assertGreater(
            float(carried.aux.metrics["memory_bridge_update_norm"].detach()),
            0.0,
        )
        self.assertFalse(torch.allclose(carried.logits, ablated.logits))
        self.assertIsNotNone(model.memory_bridge_value_projection.weight.grad)
        self.assertGreater(
            float(model.memory_bridge_value_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_actual_tier_bridge_experiment_contract(self):
        from experiments.benchmark_memory_tier_bridge_validation import (
            run_memory_tier_bridge_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_memory_tier_bridge_validation(
                output_dir=Path(tmp),
                seeds=(5,),
                train_steps=2,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertIn("multi_timescale_no_bridge", result["variants"])
        self.assertIn("semantic_procedural_bridge", result["variants"])
        bridge = result["variants"]["semantic_procedural_bridge"]
        self.assertIn("carry_accuracy", bridge)
        self.assertIn("reset_accuracy", bridge)
        self.assertIn("semantic_procedural_ablation_accuracy", bridge)
        self.assertIn("causal_ablation_drop", bridge)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
