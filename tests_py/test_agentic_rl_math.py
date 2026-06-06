import unittest

import torch

from tac_transformer import VerifierCase, build_authority_report
from tac_transformer.agentic_rl_math import (
    AgenticProofThresholds,
    AgenticTrajectoryStep,
    ScratchpadItem,
    SimulationBranch,
    agentic_promotion_decision,
    build_agentic_trajectory,
    bounded_scratchpad_update,
    commit_verified_scratchpad_items,
    cost_adjusted_rewards,
    group_relative_advantages,
    group_relative_trajectory_policy_loss,
    gspo_sequence_policy_loss,
    implicit_process_rewards,
    policy_gradient_loss,
    process_trace_distillation_loss,
    select_best_simulation_branch,
    dapo_dynamic_sampling_filter,
    basal_apical_belief_state,
    coalition_participation_metrics,
    identity_persistence_score,
    memory_link_utility,
    memory_overlap_graph,
    phase_d_agentic_reward,
    shaped_trajectory_rewards,
    trajectory_to_training_record,
    value_prediction_loss,
    verifier_reward_from_authority_report,
)


class AgenticRLMathTest(unittest.TestCase):
    def test_group_relative_advantages_are_cost_aware_and_zero_mean(self):
        rewards = torch.tensor([[1.0, 0.5, 0.0]])
        costs = torch.tensor([[0.1, 0.1, 0.6]])

        adjusted = cost_adjusted_rewards(rewards, costs, cost_weight=0.5)
        advantages = group_relative_advantages(adjusted, dim=1)

        self.assertGreater(float(adjusted[0, 0]), float(adjusted[0, 1]))
        self.assertLess(float(adjusted[0, 2]), 0.0)
        self.assertAlmostEqual(float(advantages.mean(dim=1)[0]), 0.0, places=6)
        self.assertGreater(float(advantages[0, 0]), 0.0)
        self.assertLess(float(advantages[0, 2]), 0.0)

    def test_policy_gradient_loss_moves_positive_advantage_action_up(self):
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
        actions = torch.tensor([0, 1])
        advantages = torch.tensor([1.0, -1.0])

        loss = policy_gradient_loss(logits, actions, advantages)
        loss.backward()

        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[1, 1]), 0.0)

    def test_scratchpad_is_bounded_and_commit_requires_verification(self):
        items = [
            ScratchpadItem("low", "plan", "weak", utility=0.1, confidence=0.8),
            ScratchpadItem("best", "plan", "strong", utility=0.9, confidence=0.9),
            ScratchpadItem(
                "imagined",
                "simulation",
                "hypothesis",
                utility=1.0,
                confidence=0.95,
                imagined=True,
            ),
        ]

        scratchpad = bounded_scratchpad_update([], items, budget=2)
        committed = commit_verified_scratchpad_items(
            scratchpad,
            verifier_supported_ids={"best"},
            min_confidence=0.5,
        )

        self.assertEqual(len(scratchpad), 2)
        self.assertEqual([item.item_id for item in committed], ["best"])
        self.assertNotIn("imagined", [item.item_id for item in committed])

    def test_future_simulation_selects_cost_adjusted_branch_without_committing(self):
        branches = [
            SimulationBranch("fast", ("answer",), predicted_reward=0.7, cost=0.1),
            SimulationBranch("deep", ("think", "answer"), predicted_reward=0.9, cost=0.8),
            SimulationBranch("risky", ("guess",), predicted_reward=0.95, cost=0.1, risk=0.9),
        ]

        selected = select_best_simulation_branch(
            branches,
            cost_weight=0.4,
            risk_weight=1.0,
        )
        committed = commit_verified_scratchpad_items(
            [selected.to_scratchpad_item()],
            verifier_supported_ids=set(),
        )

        self.assertEqual(selected.branch_id, "fast")
        self.assertEqual(committed, [])

    def test_process_trace_distillation_loss_weights_verified_steps(self):
        targets = torch.tensor([[0, 1]])
        verified_scores = torch.tensor([[1.0, 0.25]])
        strong_logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
        weak_logits = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]])

        strong_loss = process_trace_distillation_loss(
            strong_logits,
            targets,
            verifier_scores=verified_scores,
        )
        weak_loss = process_trace_distillation_loss(
            weak_logits,
            targets,
            verifier_scores=verified_scores,
        )

        self.assertLess(float(strong_loss), float(weak_loss))

    def test_rollout_trajectory_records_actions_state_and_adjusted_reward(self):
        trajectory = build_agentic_trajectory(
            trajectory_id="traj-1",
            steps=(
                AgenticTrajectoryStep(
                    step_index=0,
                    action="write_scratchpad",
                    action_logprob=-0.2,
                    route_id="program_2",
                    memory_read_ids=("mem_a",),
                    scratchpad_item_ids=("fact_a",),
                    verifier_score=1.0,
                    reward=0.2,
                    cost=0.1,
                ),
                AgenticTrajectoryStep(
                    step_index=1,
                    action="answer",
                    action_logprob=-0.1,
                    route_id="program_2",
                    memory_read_ids=("mem_b",),
                    scratchpad_item_ids=("fact_a", "answer"),
                    verifier_score=0.8,
                    reward=1.0,
                    cost=0.2,
                ),
            ),
            final_reward=1.0,
            cost_weight=0.5,
            metadata={"task_id": "scratchpad_copy"},
        )
        record = trajectory_to_training_record(trajectory)

        self.assertEqual(record["trajectory_id"], "traj-1")
        self.assertEqual(record["actions"], ["write_scratchpad", "answer"])
        self.assertEqual(record["route_ids"], ["program_2", "program_2"])
        self.assertEqual(record["memory_read_ids"], [["mem_a"], ["mem_b"]])
        self.assertEqual(record["scratchpad_item_ids"][-1], ["fact_a", "answer"])
        self.assertEqual(record["verifier_scores"], [1.0, 0.8])
        self.assertAlmostEqual(record["action_logprob_sum"], -0.3, places=6)
        self.assertAlmostEqual(record["total_cost"], 0.3, places=6)
        self.assertAlmostEqual(record["cost_adjusted_reward"], 0.85, places=6)
        self.assertEqual(record["metadata"]["task_id"], "scratchpad_copy")

    def test_verifier_reward_uses_authority_report_counts(self):
        report = build_authority_report(
            run_id="verifier-reward",
            events=[
                VerifierCase(
                    "ok",
                    "math",
                    expected="4",
                    observed="4",
                    authority_mode="verified_execution",
                ).verify(),
                VerifierCase(
                    "false",
                    "math",
                    expected="4",
                    observed="5",
                    authority_mode="verified_execution",
                ).verify(),
                VerifierCase(
                    "cross",
                    "math",
                    expected="7",
                    observed="7",
                    authority_mode="retrieved_evidence",
                    source_domain="history",
                ).verify(),
            ],
        )

        reward = verifier_reward_from_authority_report(
            report,
            base_reward=1.0,
            trusted_correct_bonus=0.25,
            false_authority_penalty=1.0,
            cross_domain_penalty=0.5,
        )

        self.assertAlmostEqual(reward["verifier_reward"], -0.25, places=6)
        self.assertAlmostEqual(reward["false_authority_rate"], 1.0 / 3.0, places=6)
        self.assertEqual(reward["trusted_correct_count"], 2)
        self.assertEqual(reward["false_authority_count"], 1)
        self.assertEqual(reward["cross_domain_authority_violation_count"], 1)

    def test_group_relative_trajectory_policy_loss_uses_cost_adjusted_rewards(self):
        trajectories = (
            build_agentic_trajectory(
                trajectory_id="good",
                steps=(
                    AgenticTrajectoryStep(
                        0,
                        "answer",
                        -0.2,
                        "decoder",
                        reward=1.0,
                        cost=0.1,
                    ),
                ),
                final_reward=1.0,
                cost_weight=0.1,
                metadata={"prompt_id": "same"},
            ),
            build_agentic_trajectory(
                trajectory_id="bad",
                steps=(
                    AgenticTrajectoryStep(
                        0,
                        "guess",
                        -0.2,
                        "decoder",
                        reward=0.0,
                        cost=0.5,
                    ),
                ),
                final_reward=0.0,
                cost_weight=0.1,
                metadata={"prompt_id": "same"},
            ),
        )
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)

        result = group_relative_trajectory_policy_loss(
            trajectories,
            action_logits=logits,
            actions=torch.tensor([0, 1]),
            group_ids=["same", "same"],
        )
        result["loss"].backward()

        self.assertGreater(float(result["advantages"][0]), 0.0)
        self.assertLess(float(result["advantages"][1]), 0.0)
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[1, 1]), 0.0)

    def test_dynamic_sampling_and_length_cost_shaping(self):
        short_good = build_agentic_trajectory(
            trajectory_id="short_good",
            steps=(
                AgenticTrajectoryStep(0, "answer", -0.1, "decoder", reward=1.0, cost=0.1),
            ),
            final_reward=1.0,
            cost_weight=0.0,
            metadata={"prompt_id": "mixed"},
        )
        bad = build_agentic_trajectory(
            trajectory_id="bad",
            steps=(
                AgenticTrajectoryStep(0, "guess", -0.1, "decoder", reward=0.0, cost=0.1),
            ),
            final_reward=0.0,
            cost_weight=0.0,
            metadata={"prompt_id": "mixed"},
        )
        long_good = build_agentic_trajectory(
            trajectory_id="long_good",
            steps=tuple(
                AgenticTrajectoryStep(
                    index,
                    "think" if index < 3 else "answer",
                    -0.1,
                    "decoder",
                    reward=1.0 if index == 3 else 0.0,
                    cost=0.2,
                )
                for index in range(4)
            ),
            final_reward=1.0,
            cost_weight=0.0,
            metadata={"prompt_id": "solved"},
        )
        trajectories = (short_good, bad, long_good)

        shaped = shaped_trajectory_rewards(
            trajectories,
            cost_weight=0.5,
            length_penalty=0.05,
        )
        selected = dapo_dynamic_sampling_filter(
            trajectories,
            group_ids=["mixed", "mixed", "solved"],
            success_threshold=0.5,
        )

        self.assertGreater(float(shaped[0]), float(shaped[1]))
        self.assertGreater(float(shaped[0]), float(shaped[2]))
        self.assertEqual(selected["selected_indexes"], [0, 1])
        self.assertEqual(selected["dropped_group_ids"], ["solved"])

    def test_identity_coalition_and_linkage_upgrade_support(self):
        ips = identity_persistence_score(
            torch.tensor([0.9, 0.2]),
            torch.tensor([0.8, 0.3]),
            torch.tensor([0.7, 0.4]),
        )

        self.assertGreater(float(ips[0]), 0.75)
        self.assertLess(float(ips[1]), 0.35)

        program_memory = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        overlap = memory_overlap_graph(
            program_memory,
            coactivation_window=torch.tensor(
                [
                    [0.0, 1.0, 9.0],
                    [1.0, 0.0, 1.0],
                    [9.0, 1.0, 0.0],
                ]
            ),
            tau_link=0.8,
            tau_time=2.0,
        )

        self.assertTrue(bool(overlap["link_adjacency"][0, 1]))
        self.assertFalse(bool(overlap["link_adjacency"][0, 2]))

        link = memory_link_utility(
            torch.tensor([1.0, 0.0, 0.0]),
            overlap["link_adjacency"],
            target_scores=torch.tensor([0.0, 1.0, 0.0]),
        )

        self.assertEqual(float(link["direct_utility"]), 0.0)
        self.assertGreater(float(link["linked_utility"]), 0.0)

        coalition = coalition_participation_metrics(
            torch.tensor(
                [
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ]
            )
        )

        self.assertGreater(
            float(coalition["coactivation_matrix"][0, 1]),
            float(coalition["coactivation_matrix"][0, 2]),
        )
        self.assertGreater(float(coalition["participation"][1]), 0.9)

        belief = basal_apical_belief_state(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        )

        self.assertLess(float(belief["disagreement"][0]), 1e-6)
        self.assertGreater(float(belief["disagreement"][1]), 1.5)
        self.assertEqual(belief["belief"].shape, (2, 2))

        reward = phase_d_agentic_reward(
            {
                "task_success": 1.0,
                "verification_pass": 1.0,
                "state_utility": 0.2,
                "route_utility": 0.1,
                "identity_persistence": float(ips[0]),
                "memory_link_utility": float(link["linked_utility"]),
                "coalition_utility": 0.25,
                "world_accuracy": 0.5,
                "false_authority": 0.0,
                "hypothesis_contamination": 0.0,
                "cost": 0.4,
            },
            weights={
                "task": 1.0,
                "verify": 0.5,
                "state": 1.0,
                "route": 1.0,
                "ips": 1.0,
                "link": 1.0,
                "coalition": 1.0,
                "world": 0.2,
                "false": 1.0,
                "contam": 1.0,
                "cost": 0.5,
            },
        )

        self.assertGreater(reward["reward"], 2.0)
        self.assertIn("identity_persistence", reward["positive_terms"])

    def test_dynamic_sampling_can_use_shaped_reward_success(self):
        cheap_success = build_agentic_trajectory(
            trajectory_id="cheap_success",
            steps=(AgenticTrajectoryStep(0, "answer", -0.1, "decoder", cost=0.1),),
            final_reward=1.0,
            cost_weight=0.0,
        )
        expensive_success = build_agentic_trajectory(
            trajectory_id="expensive_success",
            steps=tuple(
                AgenticTrajectoryStep(index, "think", -0.1, "decoder", cost=0.5)
                for index in range(4)
            ),
            final_reward=1.0,
            cost_weight=0.0,
        )
        shaped = shaped_trajectory_rewards(
            (cheap_success, expensive_success),
            cost_weight=1.0,
            length_penalty=0.0,
        )
        selected = dapo_dynamic_sampling_filter(
            (cheap_success, expensive_success),
            group_ids=["cost_sensitive", "cost_sensitive"],
            reward_values=shaped,
            success_threshold=0.5,
        )

        self.assertEqual(selected["selected_indexes"], [0, 1])
        self.assertEqual(selected["selected_group_ids"], ["cost_sensitive"])

    def test_sequence_process_and_value_support(self):
        current_logprobs = torch.zeros(2, 3, requires_grad=True)
        reference_logprobs = torch.zeros(2, 3)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

        sequence_result = gspo_sequence_policy_loss(
            current_logprobs,
            reference_logprobs,
            advantages,
            mask=mask,
            clip_epsilon=0.2,
        )
        sequence_result["loss"].backward()

        self.assertLess(float(current_logprobs.grad[0, 0]), 0.0)
        self.assertGreater(float(current_logprobs.grad[1, 0]), 0.0)
        self.assertEqual(sequence_result["sequence_ratios"].shape, (2,))

        process = implicit_process_rewards(
            torch.tensor([[0.2, -0.1, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            verifier_scores=torch.tensor([[1.0, 0.0, 0.5]]),
            mask=torch.tensor([[1.0, 1.0, 0.0]]),
            beta=0.5,
        )

        self.assertGreater(float(process[0, 0]), float(process[0, 1]))
        self.assertEqual(float(process[0, 2]), 0.0)

        values = torch.tensor([[0.8, 0.1, 0.0]], requires_grad=True)
        returns = torch.tensor([[1.0, 0.0, 0.0]])
        value_loss = value_prediction_loss(
            values,
            returns,
            mask=torch.tensor([[1.0, 1.0, 0.0]]),
        )
        value_loss.backward()

        self.assertGreater(float(value_loss.detach()), 0.0)
        self.assertLess(float(values.grad[0, 0]), 0.0)
        self.assertEqual(float(values.grad[0, 2]), 0.0)

    def test_agentic_promotion_decision_blocks_failed_state_and_cost_gates(self):
        decision = agentic_promotion_decision(
            {
                "carry_score": 0.27,
                "reset_score": 0.24,
                "shuffled_score": 0.28,
                "baseline_score": 0.24,
                "scratchpad_score": 0.30,
                "no_scratchpad_score": 0.29,
                "simulation_score": 0.31,
                "no_simulation_score": 0.30,
                "cost_adjusted_reward": 0.20,
                "baseline_cost_adjusted_reward": 0.25,
                "false_authority_rate": 0.0,
                "hypothesis_contamination_rate": 0.0,
                "world_error": 0.05,
                "teaching_score": 0.31,
                "no_teaching_score": 0.30,
            },
            AgenticProofThresholds(),
        )

        self.assertEqual(decision["status"], "blocked")
        self.assertFalse(decision["checks"]["carry_beats_shuffled"])
        self.assertFalse(decision["checks"]["cost_adjusted_reward_beats_baseline"])


if __name__ == "__main__":
    unittest.main()
