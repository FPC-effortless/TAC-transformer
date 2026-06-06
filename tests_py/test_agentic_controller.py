import unittest

import torch

from experiments import benchmark_agentic_controller_learning as bench
from experiments import benchmark_live_agentic_policy_adapter as live_bench
from experiments import benchmark_live_agentic_policy_training as live_train_bench
from tac_transformer import (
    AgenticScratchpadState,
    AgenticPolicyController,
    SimulationBranch,
    ScratchpadItem,
    TACConfig,
    TACTransformerLM,
    agentic_controller_supervised_loss,
    apply_agentic_scratchpad_transition,
    build_agentic_policy_features_from_tac_output,
    run_agentic_policy_controller_from_tac_output,
)


class AgenticControllerLearningTest(unittest.TestCase):
    def test_controller_learns_scratchpad_simulation_and_teaching_policies(self):
        report = bench.run_agentic_controller_learning_probe(
            example_count=48,
            train_steps=120,
            seed=13,
        )

        self.assertEqual(report["decision"]["status"], "policy_learned")
        self.assertGreaterEqual(report["controller"]["scratchpad_policy_score"], 0.95)
        self.assertGreaterEqual(report["controller"]["simulation_policy_score"], 0.95)
        self.assertGreaterEqual(report["controller"]["teaching_policy_score"], 0.95)
        self.assertEqual(report["controller"]["hypothesis_contamination_rate"], 0.0)
        self.assertGreater(
            report["controller"]["scratchpad_policy_score"],
            report["controls"]["no_scratchpad_score"],
        )
        self.assertGreater(
            report["controller"]["simulation_policy_score"],
            report["controls"]["no_simulation_score"],
        )
        self.assertGreater(
            report["controller"]["teaching_policy_score"],
            report["controls"]["no_teaching_score"],
        )

    def test_policy_controller_accepts_live_tac_output_features(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        tac_output = model(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
        branches = (
            SimulationBranch("safe", ("answer",), 0.7, 0.1, risk=0.0),
            SimulationBranch("deep", ("think", "answer"), 0.9, 0.5, risk=0.1),
            SimulationBranch("risky", ("guess",), 0.99, 0.1, risk=0.9),
        )

        features = build_agentic_policy_features_from_tac_output(
            tac_output,
            branches=branches,
            scratchpad_slots=3,
        )
        controller = AgenticPolicyController()
        policy = run_agentic_policy_controller_from_tac_output(
            controller,
            tac_output,
            branches=branches,
            scratchpad_slots=3,
        )
        loss = (
            policy["scratchpad_logits"].mean()
            + policy["simulation_logits"].mean()
            + policy["process_logits"].mean()
        )
        loss.backward()

        self.assertEqual(features.scratchpad_features.shape, (2, 3, 8))
        self.assertEqual(features.simulation_features.shape, (2, 3, 5))
        self.assertEqual(features.context_features.shape, (2, 4))
        self.assertEqual(policy["scratchpad_logits"].shape, (2, 3))
        self.assertEqual(policy["simulation_logits"].shape, (2, 3))
        self.assertEqual(policy["process_logits"].shape, (2, 4, 4))
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum()), 0.0)

    def test_live_adapter_benchmark_reports_connected_policy_features(self):
        report = live_bench.run_live_agentic_policy_adapter_probe(seed=17)

        self.assertEqual(report["decision"]["status"], "live_features_connected")
        self.assertEqual(report["features"]["scratchpad_shape"], [2, 3, 8])
        self.assertEqual(report["features"]["simulation_shape"], [2, 3, 5])
        self.assertEqual(report["features"]["context_shape"], [2, 4])
        self.assertGreater(report["gradient_flow"]["token_embedding_grad_abs_sum"], 0.0)

    def test_live_policy_training_preserves_frozen_tac_capability(self):
        report = live_train_bench.run_live_agentic_policy_training_probe(
            example_count=48,
            train_steps=120,
            seed=19,
        )

        self.assertEqual(
            report["decision"]["status"],
            "live_policy_trained_capability_preserved",
        )
        self.assertLess(report["training"]["final_loss"], report["training"]["initial_loss"])
        self.assertGreaterEqual(report["policy"]["scratchpad_policy_score"], 0.95)
        self.assertGreaterEqual(report["policy"]["simulation_policy_score"], 0.95)
        self.assertGreaterEqual(report["policy"]["teaching_policy_score"], 0.95)
        self.assertLessEqual(report["capability_preservation"]["max_logit_drift"], 1e-8)
        self.assertLessEqual(report["capability_preservation"]["eval_loss_drift"], 1e-8)

    def test_controller_value_head_trains_from_context_returns(self):
        controller = AgenticPolicyController()
        scratchpad_features = torch.randn(3, 3, 8)
        simulation_features = torch.randn(3, 3, 5)
        context_features = torch.randn(3, 4)

        outputs = controller(
            scratchpad_features=scratchpad_features,
            simulation_features=simulation_features,
            context_features=context_features,
        )
        self.assertEqual(outputs["value"].shape, (3,))

        losses = agentic_controller_supervised_loss(
            outputs,
            scratchpad_targets=torch.ones(3, 3),
            simulation_targets=torch.zeros(3, dtype=torch.long),
            process_targets=torch.zeros(3, 4, dtype=torch.long),
            verifier_scores=torch.ones(3, 4),
            value_targets=torch.tensor([1.0, 0.5, -0.5]),
            value_weight=0.5,
        )
        losses["loss"].backward()

        self.assertIn("value_loss", losses)
        self.assertGreater(float(losses["value_loss"]), 0.0)
        self.assertIsNotNone(controller.value_head[-1].weight.grad)
        self.assertGreater(float(controller.value_head[-1].weight.grad.abs().sum()), 0.0)

    def test_scratchpad_state_transition_commits_only_verified_controller_choices(self):
        state = AgenticScratchpadState.empty(budget=2)
        candidates = (
            ScratchpadItem(
                "left",
                "observation",
                "2",
                utility=0.8,
                confidence=0.95,
            ),
            ScratchpadItem(
                "right",
                "observation",
                "7",
                utility=0.8,
                confidence=0.95,
            ),
            ScratchpadItem(
                "imagined",
                "simulation",
                "9",
                utility=1.0,
                confidence=0.99,
                imagined=True,
            ),
        )

        next_state, report = apply_agentic_scratchpad_transition(
            state,
            candidates,
            commit_logits=torch.tensor([8.0, 8.0, 8.0]),
            verifier_supported_ids={"left", "right"},
        )

        self.assertEqual([item.item_id for item in next_state.items], ["left", "right"])
        self.assertTrue(all(item.verified for item in next_state.items))
        self.assertEqual(next_state.step, 1)
        self.assertEqual(report["committed_ids"], ["left", "right"])
        self.assertEqual(report["rejected_ids"], ["imagined"])
        self.assertEqual(report["hypothesis_contamination_rate"], 0.0)
        self.assertLessEqual(len(next_state.items), next_state.budget)


if __name__ == "__main__":
    unittest.main()
