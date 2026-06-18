import json
import tempfile
import unittest
from pathlib import Path

import torch

from tac_transformer import TACConfig, TACTransformerLM


class BDHTACAdaptationTests(unittest.TestCase):
    def test_sparse_positive_program_activation_type_creates_exact_zeros(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=8,
            program_activation_type="relu",
            activation_l1_weight=0.1,
        )
        model = TACTransformerLM(config)
        field = model.blocks[0].identity_field

        activations = field._program_activations(
            torch.tensor([[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]])
        )
        output = model(torch.tensor([[1, 2, 3, 4]]))

        self.assertTrue((activations >= 0.0).all())
        self.assertGreater(
            float((activations == 0.0).float().mean()),
            0.25,
        )
        self.assertIn("activation_density", output.aux.metrics)

    def test_hebbian_outer_memory_write_updates_selected_program_slots(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            routing_type="base_semantic",
            routing_top_k=2,
            memory_write_type="hebbian_outer",
            state_decay=0.0,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        output = model(input_ids)
        memory_norm = output.identity_states[0].program_memory.norm(dim=-1)
        selected = output.aux.token_selected_program_mask.bool().any(dim=1)

        self.assertGreater(float(memory_norm[selected].mean()), 0.0)
        self.assertLess(float(memory_norm[~selected].max()), 1e-6)
        self.assertIn("hebbian_write_strength", output.aux.metrics)

    def test_bdh_benchmark_runs_all_adaptation_probes(self):
        from experiments.benchmark_bdh_tac_adaptations import run_bdh_tac_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            result = run_bdh_tac_benchmark(
                output_dir=Path(tmp),
                seeds=(3,),
                batch_size=2,
                seq_len=6,
                vocab_size=48,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["research_basis"]["arxiv_id"], "2509.26507")
        self.assertIn("hebbian_working_memory", result["adaptations"])
        self.assertIn("sparse_positive_activations", result["adaptations"])
        self.assertIn("stateful_moe_programs", result["adaptations"])
        self.assertIn("modular_graph_topology", result["adaptations"])
        self.assertIn("state_space_batched_recurrence", result["adaptations"])
        self.assertIn("memory_as_state", result["adaptations"])
        self.assertIn("interpretability_constraints", result["adaptations"])
        self.assertGreater(
            result["adaptations"]["hebbian_working_memory"]["selected_memory_norm"],
            0.0,
        )
        self.assertLess(
            result["adaptations"]["sparse_positive_activations"]["relu_density"],
            result["adaptations"]["sparse_positive_activations"]["sigmoid_density"],
        )
        self.assertGreater(
            result["adaptations"]["stateful_moe_programs"]["continued_route_agreement"],
            result["adaptations"]["stateful_moe_programs"]["fresh_route_agreement"],
        )
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
