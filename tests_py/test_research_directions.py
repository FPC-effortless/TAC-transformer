import unittest

import torch

from tac_transformer import ContentWritePolicy
from tac_transformer.research_directions import (
    EFFICIENCY_RESEARCH_VARIANTS,
    OBJECTIVE_RESEARCH_VARIANTS,
    computation_prediction_loss,
    format_research_directions_markdown,
    latent_state_prediction_loss,
    macro_program_compression_stats,
    predictive_coding_loss,
    program_useful_contrastive_loss,
    route_reconstruction_loss,
    summarize_efficiency_research,
    summarize_objective_research,
)
from experiments.benchmark_tac_research_directions import (
    build_objective_heads,
    should_update_decode_memory,
    write_policy_from_settings,
)
from experiments.benchmark_identity_decode_cache import (
    program_switch_fraction,
    summarize_identity_decode_cache_profile,
)
from experiments.benchmark_route_reconstruct_diagnostic import (
    classify_route_reconstruct_diagnostic,
    counterfactual_route_reconstruction_stats,
    summarize_counterfactual_rows,
)
from experiments.benchmark_program_contrastive_refinement import (
    B1_VARIANTS,
    annealed_temperature,
    hard_negative_margin_loss,
    summarize_b1_rows,
)
from experiments.benchmark_program_memory_write_diagnostic import (
    classify_write_diagnostic,
    gini,
    normalized_entropy,
    summarize_write_diagnostic_rows,
    summarize_write_rows,
)
from kaggle.benchmark_inference import (
    measure_peak_memory_bytes,
    parse_decode_policies,
    profile_label,
)


class ResearchDirectionsTest(unittest.TestCase):
    def test_objective_and_efficiency_variants_cover_deferred_tracks(self):
        self.assertIn("ntp_reference", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("run5_regularized_mi", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("latent_state", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("predictive_coding", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("program_contrastive", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("route_reconstruct", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("computation_prediction", OBJECTIVE_RESEARCH_VARIANTS)
        self.assertIn("content_every_4", EFFICIENCY_RESEARCH_VARIANTS)
        self.assertIn("event_error_update", EFFICIENCY_RESEARCH_VARIANTS)
        self.assertIn("no_content_updates", EFFICIENCY_RESEARCH_VARIANTS)
        self.assertIn("masked_prefill_query_skip", EFFICIENCY_RESEARCH_VARIANTS)

    def test_efficiency_variants_expose_write_policy_surface(self):
        self.assertEqual(
            write_policy_from_settings(EFFICIENCY_RESEARCH_VARIANTS["full_update"]),
            ContentWritePolicy.DENSE,
        )
        self.assertEqual(
            write_policy_from_settings(EFFICIENCY_RESEARCH_VARIANTS["no_content_updates"]),
            ContentWritePolicy.DISABLED,
        )
        self.assertEqual(
            write_policy_from_settings(
                EFFICIENCY_RESEARCH_VARIANTS["masked_prefill_query_skip"]
            ),
            ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP,
        )
        self.assertFalse(
            should_update_decode_memory(
                EFFICIENCY_RESEARCH_VARIANTS["masked_prefill_query_skip"],
                token_index=0,
                previous_loss=None,
            )
        )
        self.assertFalse(
            should_update_decode_memory(
                {"write_policy": ContentWritePolicy.DECODE_STATE_SKIP.value},
                token_index=0,
                previous_loss=None,
            )
        )

    def test_inference_profiler_reports_peak_memory_on_cpu(self):
        value = measure_peak_memory_bytes(lambda: [0] * 64, device=torch.device("cpu"))

        self.assertGreaterEqual(value, 0)

    def test_inference_profiler_parses_decode_policy_dimension(self):
        self.assertEqual(
            parse_decode_policies("query_skip, decode_state_skip"),
            [ContentWritePolicy.QUERY_SKIP, ContentWritePolicy.DECODE_STATE_SKIP],
        )
        self.assertEqual(
            profile_label("current_best", ContentWritePolicy.DECODE_STATE_SKIP),
            "current_best:decode_state_skip",
        )
        self.assertEqual(profile_label("vanilla_matched", None), "vanilla_matched")

    def test_identity_decode_cache_summary_reports_speedup_ceiling_and_switch_rate(self):
        summary = summarize_identity_decode_cache_profile(
            {
                "decode_seconds": 10.0,
                "decode_tokens_per_second": 100.0,
                "identity_field_seconds": 3.0,
                "identity_field_calls_per_iter": 8,
                "assignment_trace": [[1, 1], [1, 2], [1, 2]],
            }
        )

        self.assertAlmostEqual(summary["identity_field_fraction"], 0.3)
        self.assertAlmostEqual(summary["identity_cache_speedup_ceiling"], 1 / 0.7)
        self.assertAlmostEqual(summary["program_switch_fraction"], 0.25)
        self.assertEqual(summary["cache_safety_decision"], "diagnostic_only")
        self.assertAlmostEqual(program_switch_fraction([[1], [1], [2]]), 0.5)

    def test_route_reconstruct_counterfactual_stats_find_best_program(self):
        decoder = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            decoder.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                )
            )
        routes = torch.tensor([[[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]]])
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        stats = counterfactual_route_reconstruction_stats(routes, hidden, decoder)

        self.assertEqual(stats["tokens"], 2)
        self.assertAlmostEqual(stats["routed_is_best_fraction"], 0.5)
        self.assertGreater(stats["mean_routed_minus_best"], 0.0)
        summary = summarize_counterfactual_rows([stats, stats])
        self.assertEqual(summary["tokens"], 4)
        self.assertAlmostEqual(
            summary["mean_routed_minus_best"],
            stats["mean_routed_minus_best"],
        )

    def test_route_reconstruct_diagnostic_classifies_blockers(self):
        blocked = classify_route_reconstruct_diagnostic(
            {"routed_is_best_fraction": 1.0, "mean_routed_minus_best": 0.0, "mean_other_minus_routed": 1.0},
            {"max_route_grad_norm": 0.0},
        )
        misaligned = classify_route_reconstruct_diagnostic(
            {"routed_is_best_fraction": 0.1, "mean_routed_minus_best": 0.01, "mean_other_minus_routed": -0.01},
            {"max_route_grad_norm": 0.1},
        )

        self.assertEqual(blocked["verdict"], "gradient_flow_blocked")
        self.assertEqual(misaligned["verdict"], "routing_not_functionally_aligned")

    def test_b1_hard_negative_loss_rewards_target_over_close_negative(self):
        self.assertIn("task_conditioned_memsep_0p1", B1_VARIANTS)
        self.assertIn("task_conditioned_memsep_1p0", B1_VARIANTS)
        reconstruction_losses = torch.tensor([[[0.1, 0.2, 1.0]]])
        weak_log_probs = torch.log_softmax(torch.tensor([[[0.0, 0.0, 0.0]]]), dim=-1)
        strong_log_probs = torch.log_softmax(torch.tensor([[[2.0, 0.0, -1.0]]]), dim=-1)

        self.assertGreater(
            float(hard_negative_margin_loss(weak_log_probs, reconstruction_losses)),
            float(hard_negative_margin_loss(strong_log_probs, reconstruction_losses)),
        )
        self.assertGreater(annealed_temperature(0, 5), annealed_temperature(4, 5))

    def test_b1_summary_prefers_passing_aligned_variant(self):
        rows = [
            {
                "variant": "weak",
                "decision": {"passes_b1_local_gate": False},
                "counterfactual_reconstruction": {"routed_is_best_fraction": 0.1},
                "mean_program_memory_cosine": 0.9,
                "mean_program_embedding_offdiag_cosine": 0.9,
            },
            {
                "variant": "strong",
                "decision": {"passes_b1_local_gate": True},
                "counterfactual_reconstruction": {"routed_is_best_fraction": 0.3},
                "mean_program_memory_cosine": 0.7,
                "mean_program_embedding_offdiag_cosine": 0.7,
            },
        ]

        summary = summarize_b1_rows(rows)

        self.assertEqual(summary["recommendation"]["variant"], "strong")

    def test_program_memory_write_diagnostic_summarizes_load_health(self):
        uniform = torch.ones(4)
        concentrated = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rows = [
            {
                "program_memory_cosine": 0.9,
                "memory_norm_mean": 1.0,
                "memory_norm_max": 2.0,
                "dead_program_fraction": 0.0,
                "selected_load_entropy": normalized_entropy(uniform),
                "selected_load_gini": gini(uniform),
                "write_frequency_entropy": normalized_entropy(concentrated),
                "write_frequency_gini": gini(concentrated),
                "age_mean": 1.0,
                "age_gini": 0.0,
                "per_program_memory_norm": [1.0, 1.0, 1.0, 1.0],
                "per_program_write_frequency": [1.0, 0.0, 0.0, 0.0],
                "per_program_selected_load": [0.25, 0.25, 0.25, 0.25],
            }
        ]

        summary = summarize_write_rows(rows)
        decision = classify_write_diagnostic(summary)

        self.assertAlmostEqual(summary["selected_load_entropy"], 1.0)
        self.assertGreater(summary["write_frequency_gini"], 0.0)
        self.assertEqual(decision["verdict"], "write_allocation_concentrated")
        dead = classify_write_diagnostic(
            {
                "program_memory_cosine": 0.0,
                "dead_program_fraction": 0.75,
                "write_frequency_entropy": 0.9,
                "selected_load_entropy": 0.9,
            }
        )
        self.assertEqual(dead["verdict"], "memory_dead_or_underwritten")

    def test_program_memory_write_summary_does_not_recommend_dead_low_cosine_row(self):
        rows = [
            {
                "variant": "dead_low_cosine",
                "decision": {"verdict": "memory_dead_or_underwritten"},
                "eval_write_stats": {
                    "program_memory_cosine": 0.0,
                    "dead_program_fraction": 0.75,
                    "write_frequency_entropy": 0.5,
                },
            },
            {
                "variant": "alive_collapsed",
                "decision": {"verdict": "memory_update_collapsed_despite_broad_writes"},
                "eval_write_stats": {
                    "program_memory_cosine": 0.9,
                    "dead_program_fraction": 0.0,
                    "write_frequency_entropy": 1.0,
                },
            },
        ]

        summary = summarize_write_diagnostic_rows(rows)

        self.assertIsNone(summary["recommendation"])
        self.assertEqual(summary["best_observed"]["variant"], "dead_low_cosine")

    def test_deferred_objective_losses_are_finite_and_differentiable(self):
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        routes = torch.softmax(torch.randn(2, 5, 4), dim=-1)
        categories = torch.tensor([0, 1])
        latent_head = torch.nn.Linear(8, 8)
        error_head = torch.nn.Linear(8, 8)
        route_decoder = torch.nn.Linear(4, 8)
        computation_head = torch.nn.Linear(8, 4)

        loss = (
            latent_state_prediction_loss(hidden, latent_head)
            + predictive_coding_loss(hidden, error_head)
            + program_useful_contrastive_loss(routes, categories)
            + route_reconstruction_loss(routes, hidden, route_decoder)
            + computation_prediction_loss(hidden, routes, computation_head)
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(hidden.grad)

    def test_program_contrastive_rewards_useful_not_arbitrary_difference(self):
        categories = torch.tensor([0, 0, 1, 1])
        useful_routes = torch.tensor(
            [
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.8, 0.2], [0.9, 0.1]],
                [[0.1, 0.9], [0.1, 0.9]],
                [[0.2, 0.8], [0.1, 0.9]],
            ],
            dtype=torch.float32,
        )
        collapsed_routes = torch.full((4, 2, 2), 0.5)

        useful_loss = program_useful_contrastive_loss(useful_routes, categories)
        collapsed_loss = program_useful_contrastive_loss(collapsed_routes, categories)

        self.assertLess(float(useful_loss), float(collapsed_loss))

    def test_macro_program_stats_detect_repeated_sequences(self):
        assignments = torch.tensor(
            [
                [1, 2, 3, 1, 2, 3],
                [1, 2, 3, 4, 1, 2],
            ]
        )

        stats = macro_program_compression_stats(assignments, max_order=3)

        self.assertEqual(stats["best_order"], 3)
        self.assertEqual(stats["top_sequence"], [1, 2, 3])
        self.assertGreater(stats["macro_savings_upper_bound"], 0.0)

    def test_summaries_rank_and_format_results(self):
        objective_rows = [
            {
                "variant": "ntp_reference",
                "loss_improvement": 0.5,
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 5.5, "accuracy": 0.1, "program_memory_cosine": 0.9},
                "route_specialization": {"selected_mi_bits": 0.01, "activation_mi_bits": 0.02},
                "train": {"tokens_per_second": 100.0},
            },
            {
                "variant": "latent_state",
                "loss_improvement": 0.7,
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 5.3, "accuracy": 0.2, "program_memory_cosine": 0.8},
                "route_specialization": {"selected_mi_bits": 0.03, "activation_mi_bits": 0.04},
                "train": {"tokens_per_second": 90.0},
            },
        ]
        efficiency_rows = [
            {"mode": "full_update", "loss": 5.0, "accuracy": 0.1, "tokens_per_second": 50.0, "update_fraction": 1.0},
            {"mode": "serving_no_aux", "loss": 5.0, "accuracy": 0.1, "tokens_per_second": 100.0, "update_fraction": 1.0},
        ]

        result = {
            "objective_summary": summarize_objective_research(objective_rows),
            "efficiency_summary": summarize_efficiency_research(efficiency_rows),
        }
        markdown = format_research_directions_markdown(result)

        self.assertEqual(result["objective_summary"]["recommendation"]["variant"], "latent_state")
        self.assertEqual(result["efficiency_summary"]["recommendation"]["mode"], "serving_no_aux")
        self.assertIn("latent_state", markdown)
        self.assertIn("serving_no_aux", markdown)

    def test_benchmark_helpers_build_heads_and_event_updates(self):
        config = type("Config", (), {"d_model": 8, "n_programs": 4})()
        settings = {
            "latent_state_weight": 0.1,
            "predictive_coding_weight": 0.1,
            "route_reconstruct_weight": 0.1,
            "computation_prediction_weight": 0.1,
        }

        heads = build_objective_heads(config, settings)
        event_settings = {
            "update_content_memory": True,
            "decode_update_interval": -1,
            "event_loss_threshold": 4.0,
        }

        self.assertIn("latent_predictor", heads)
        self.assertIn("error_predictor", heads)
        self.assertIn("route_decoder", heads)
        self.assertIn("computation_predictor", heads)
        self.assertTrue(
            should_update_decode_memory(
                event_settings,
                token_index=0,
                previous_loss=None,
            )
        )
        self.assertFalse(
            should_update_decode_memory(
                event_settings,
                token_index=1,
                previous_loss=1.0,
            )
        )
        self.assertTrue(
            should_update_decode_memory(
                event_settings,
                token_index=1,
                previous_loss=5.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
