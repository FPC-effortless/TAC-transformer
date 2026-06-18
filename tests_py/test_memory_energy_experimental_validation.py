import json
import tempfile
import unittest
from pathlib import Path

import torch

from tac_transformer import TACConfig, TACTransformerLM


class MemoryEnergyExperimentalValidationTests(unittest.TestCase):
    def test_multi_timescale_identity_state_is_model_facing(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            memory_system_type="multi_timescale",
        )
        model = TACTransformerLM(config)
        output = model(torch.tensor([[1, 2, 3, 4]]))
        state = output.identity_states[0]

        self.assertEqual(state.working_state.shape, (1, config.n_programs, config.d_model))
        self.assertEqual(state.episodic_state.shape, (1, config.n_programs, config.d_model))
        self.assertEqual(state.semantic_state.shape, (1, config.n_programs, config.d_model))
        self.assertEqual(state.procedural_state.shape, (1, config.n_programs, config.d_model))
        self.assertEqual(state.memory_confidence.shape, (1, config.n_programs))
        self.assertIn("multi_timescale_memory_mass", output.aux.metrics)
        self.assertIn("memory_confidence", output.aux.metrics)

    def test_actual_training_experiment_contract(self):
        from experiments.benchmark_memory_energy_experimental_validation import (
            run_experimental_memory_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_experimental_memory_validation(
                output_dir=Path(tmp),
                seeds=(3,),
                train_steps=2,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertIn("flat_control", result["variants"])
        self.assertIn("multi_timescale", result["variants"])
        self.assertIn("carry_accuracy", result["variants"]["multi_timescale"])
        self.assertIn("reset_accuracy", result["variants"]["multi_timescale"])
        self.assertIn("semantic_procedural_ablation_accuracy", result["variants"]["multi_timescale"])
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
