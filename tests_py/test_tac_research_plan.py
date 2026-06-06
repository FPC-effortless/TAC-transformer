import unittest

from tac_transformer.research_plan import (
    TAC_CONTROL_V1_CONFIG,
    aggregate_phase_b_seed_results,
    aggregate_phase_c_identity_stability_results,
    aggregate_phase_d_benchmark_results,
    audit_tac_control_v1,
    build_phase_b_kaggle_kernel,
    build_phase_b_replication_plan,
    build_phase_c_identity_stability_protocol,
    build_phase_d_benchmark_protocol,
    format_phase_c_identity_stability_markdown,
    format_phase_d_benchmark_results_markdown,
    format_tac_research_plan_markdown,
    format_phase_b_seed_results_markdown,
    summarize_phase_c_identity_seed,
    summarize_phase_b_seed_result,
)


class TACResearchPlanTests(unittest.TestCase):
    def test_phase_a_audit_freezes_resolved_run5b_config(self):
        manifest = {
            "config": {
                "identity_attention_type": "identity_first",
                "memory_read_type": "content_addressed",
                "content_read_gate_type": "synthesis",
                "content_read_steps": 2,
                "program_memory_update_type": "program_conditioned",
                "memory_allocation_type": "creb",
                "memory_allocation_k": 6,
                "memory_separation_weight": 0.1,
                "routing_type": "base_semantic",
                "routing_top_k": 2,
            },
            "category_route_weight": 0.1,
            "category_route_objective": "mi",
            "precision": "fp32",
        }
        summary = {
            "completed_steps": 10000,
            "target_steps": 10000,
            "best_eval_loss": 0.1672,
            "latest_metrics": {
                "tokens_seen": 183600000,
                "content_addressed_hit": 0.1874,
                "content_synthesis_gate": 0.5073,
                "program_memory_cosine": 0.0031,
                "optimization_health": {"status": "passed"},
                "eval": {
                    "loss": 0.1672,
                    "accuracy": 0.9414,
                    "content_addressed_hit": 0.1720,
                    "content_synthesis_gate": 0.5098,
                    "program_memory_cosine": 0.0025,
                },
            },
        }
        external_validation = {
            "specialization": {
                "mi_bits": 0.2821,
                "max_knockout_loss_delta": 0.3401,
                "max_knockout_selectivity_span": 0.1868,
            }
        }

        audit = audit_tac_control_v1(
            manifest,
            summary,
            external_validation=external_validation,
        )

        self.assertEqual(audit["decision"]["status"], "freeze_ready")
        self.assertEqual(audit["reference"]["name"], "TAC-Control-v1")
        self.assertEqual(audit["reference"]["checkpoint_step"], 10000)
        self.assertEqual(audit["reference"]["tokens_seen"], 183600000)
        self.assertAlmostEqual(audit["metrics"]["program_memory_cosine"], 0.0025)
        self.assertTrue(audit["config"]["memory_read_type"]["passes"])
        self.assertTrue(audit["config"]["identity_attention_type"]["passes"])
        self.assertEqual(
            audit["resolved_gaps"]["content_addressed_store"]["status"],
            "resolved",
        )
        self.assertEqual(
            audit["resolved_gaps"]["identity_first_path"]["status"],
            "resolved",
        )
        self.assertIn("multi_hop", audit["open_gaps"])

    def test_phase_b_plan_uses_frozen_config_and_seed_gates(self):
        plan = build_phase_b_replication_plan(seeds=(11, 23, 37))

        self.assertEqual(plan["phase"], "B")
        self.assertEqual([run["seed"] for run in plan["runs"]], [11, 23, 37])
        self.assertEqual(
            plan["frozen_config"]["program_memory_update_type"],
            "program_conditioned",
        )
        self.assertEqual(plan["success_criteria"]["program_memory_cosine_max"], 0.25)
        self.assertEqual(plan["success_criteria"]["selected_route_mi_min"], 0.15)
        self.assertIn("--memory-read-type content_addressed", plan["runs"][0]["command"])
        self.assertIn("--seed 11", plan["runs"][0]["command"])

    def test_phase_d_protocol_demands_consequence_over_vanilla(self):
        protocol = build_phase_d_benchmark_protocol()

        task_names = {
            task["id"]
            for suite in protocol["suites"]
            for task in suite["tasks"]
        }
        control_names = {control["id"] for control in protocol["controls"]}

        self.assertIn("multi_hop_chain_retrieval", task_names)
        self.assertIn("long_context_retrieval_4096", task_names)
        self.assertIn("tool_selection", task_names)
        self.assertIn("parameter_matched_vanilla", control_names)
        self.assertIn("tac_shuffled_state", control_names)
        self.assertEqual(protocol["decision_gate"], "TAC > parameter-matched vanilla")

    def test_phase_c_protocol_documents_identity_alignment_requirements(self):
        protocol = build_phase_c_identity_stability_protocol()

        self.assertEqual(protocol["phase"], "C")
        self.assertIn("memory_vector", protocol["alignment_components"])
        self.assertIn("selected_route_distribution", protocol["alignment_components"])
        self.assertIn("knockout_profile", protocol["alignment_components"])
        self.assertEqual(protocol["success_criteria"]["min_seed_count"], 2)

    def test_phase_b_kaggle_kernel_uses_frozen_seed_command(self):
        kernel = build_phase_b_kaggle_kernel(
            seed=11,
            code_dataset="jeffkolo/tac-run5b-capability-code-2026-06-04",
            data_dataset="jeffkolo/tac-run5b-capability-data-2026-06-03",
        )

        self.assertEqual(
            kernel["metadata"]["id"],
            "jeffkolo/tac-control-v1-phase-b-seed-11-20k",
        )
        self.assertIn(
            "jeffkolo/tac-run5b-capability-code-2026-06-04",
            kernel["metadata"]["dataset_sources"],
        )
        self.assertIn('"--seed"', kernel["script"])
        self.assertIn('"11"', kernel["script"])
        self.assertIn('"--memory-read-type"', kernel["script"])
        self.assertIn('"content_addressed"', kernel["script"])
        self.assertIn("tac_control_v1_seed_11", kernel["script"])

    def test_phase_b_seed_summary_requires_knockout_evidence(self):
        final_summary = self._phase_b_final_summary(mi_bits=0.24, run_knockouts=False)
        seed = summarize_phase_b_seed_result(
            seed=11,
            final_summary=final_summary,
            metrics_rows=[final_summary["latest_metrics"]],
        )

        self.assertEqual(seed["status"], "pending_knockout")
        self.assertTrue(seed["gates"]["eval_accuracy"]["passes"])
        self.assertTrue(seed["gates"]["program_memory_cosine"]["passes"])
        self.assertTrue(seed["gates"]["selected_route_mi"]["passes"])
        self.assertEqual(seed["gates"]["max_knockout_loss_delta"]["status"], "pending")
        self.assertIn("knockout", " ".join(seed["evidence_gaps"]))

    def test_phase_b_aggregation_promotes_only_complete_seed_evidence(self):
        passing_seed = summarize_phase_b_seed_result(
            seed=11,
            final_summary=self._phase_b_final_summary(mi_bits=0.24, run_knockouts=False),
            metrics_rows=[self._phase_b_metrics()],
            specialization_report=self._phase_b_knockout_report(mi_bits=0.24),
        )
        pending_seed = summarize_phase_b_seed_result(
            seed=23,
            final_summary=self._phase_b_final_summary(mi_bits=0.19, run_knockouts=False),
            metrics_rows=[self._phase_b_metrics()],
        )

        result = aggregate_phase_b_seed_results([passing_seed, pending_seed])

        self.assertEqual(passing_seed["status"], "pass")
        self.assertEqual(result["decision"]["status"], "pending")
        self.assertEqual(result["summary"]["passed_seed_count"], 1)
        self.assertEqual(result["summary"]["pending_seed_count"], 1)
        self.assertFalse(result["decision"]["ready_for_phase_d"])

        markdown = format_phase_b_seed_results_markdown(result)
        self.assertIn("Seed 11", markdown)
        self.assertIn("pending_knockout", markdown)

    def test_phase_c_identity_stability_blocks_until_phase_b_ready(self):
        seed = summarize_phase_c_identity_seed(
            seed=11,
            specialization_report=self._phase_c_specialization_report(),
        )

        result = aggregate_phase_c_identity_stability_results(
            [seed],
            phase_b_decision={"ready_for_phase_d": False},
        )

        self.assertEqual(result["decision"]["status"], "blocked_by_phase_b")
        self.assertFalse(result["decision"]["passes_identity_stability_gate"])

    def test_phase_c_identity_stability_aligns_permuted_program_roles(self):
        seed_11 = summarize_phase_c_identity_seed(
            seed=11,
            specialization_report=self._phase_c_specialization_report(),
        )
        seed_23 = summarize_phase_c_identity_seed(
            seed=23,
            specialization_report=self._phase_c_specialization_report(
                program_order=(1, 0),
            ),
        )

        result = aggregate_phase_c_identity_stability_results(
            [seed_11, seed_23],
            phase_b_decision={"ready_for_phase_d": True},
            min_alignment_similarity=0.80,
        )

        self.assertEqual(result["decision"]["status"], "pass")
        self.assertTrue(result["decision"]["passes_identity_stability_gate"])
        self.assertGreaterEqual(result["alignment"]["min_similarity"], 0.80)
        self.assertEqual(
            result["pairwise_alignments"][0]["matches"][0]["reference_program"],
            0,
        )
        self.assertEqual(
            result["pairwise_alignments"][0]["matches"][0]["candidate_program"],
            1,
        )
        self.assertTrue(result["component_coverage"]["memory"])
        self.assertTrue(result["component_coverage"]["route"])
        self.assertTrue(result["component_coverage"]["knockout"])

        markdown = format_phase_c_identity_stability_markdown(result)
        self.assertIn("Phase C Identity Stability", markdown)
        self.assertIn("Seed 23", markdown)

    def test_phase_d_aggregate_blocks_until_phase_b_ready(self):
        result = aggregate_phase_d_benchmark_results(
            self._phase_d_rows(),
            phase_b_decision={"ready_for_phase_d": False},
        )

        self.assertEqual(result["decision"]["status"], "blocked_by_phase_b")
        self.assertFalse(result["decision"]["ready_for_phase_d"])

    def test_phase_d_aggregate_requires_tac_to_beat_parameter_matched_vanilla(self):
        result = aggregate_phase_d_benchmark_results(
            self._phase_d_rows(),
            phase_b_decision={"ready_for_phase_d": True},
        )

        self.assertEqual(result["decision"]["status"], "pass")
        self.assertTrue(result["decision"]["passes_decision_gate"])
        self.assertGreater(
            result["families"]["memory_intensive"]["tac_advantage"],
            0.0,
        )
        self.assertGreater(result["families"]["agentic"]["tac_advantage"], 0.0)

        markdown = format_phase_d_benchmark_results_markdown(result)
        self.assertIn("Phase D Benchmark Results", markdown)
        self.assertIn("multi_hop_chain_retrieval", markdown)
        self.assertIn("parameter_matched_vanilla", markdown)

    def test_markdown_summarizes_next_stage_contract(self):
        audit = audit_tac_control_v1(
            {
                "config": TAC_CONTROL_V1_CONFIG,
                "category_route_weight": 0.1,
                "category_route_objective": "mi",
                "precision": "fp32",
            },
            {
                "completed_steps": 10000,
                "latest_metrics": {
                    "tokens_seen": 183600000,
                    "content_addressed_hit": 0.18,
                    "content_synthesis_gate": 0.50,
                    "program_memory_cosine": 0.0031,
                    "optimization_health": {"status": "passed"},
                    "eval": {"loss": 0.1672, "accuracy": 0.9414},
                },
            },
            external_validation={
                "specialization": {
                    "mi_bits": 0.2821,
                    "max_knockout_selectivity_span": 0.1868,
                }
            },
        )
        markdown = format_tac_research_plan_markdown(
            audit,
            build_phase_b_replication_plan(),
            build_phase_c_identity_stability_protocol(),
            build_phase_d_benchmark_protocol(),
        )

        self.assertIn("TAC-Control-v1", markdown)
        self.assertIn("content_addressed", markdown)
        self.assertIn("Phase B", markdown)
        self.assertIn("Phase C", markdown)
        self.assertIn("Phase D", markdown)
        self.assertIn("multi-hop", markdown)

    def _phase_b_metrics(self):
        return {
            "step": 10000,
            "tokens_seen": 183600000,
            "program_memory_cosine": 0.04,
            "tokens_per_second": 6800.0,
            "optimization_health": {"status": "passed"},
            "eval": {
                "loss": 0.16,
                "accuracy": 0.94,
                "program_memory_cosine": 0.03,
            },
        }

    def _phase_b_final_summary(self, *, mi_bits, run_knockouts):
        return {
            "completed_steps": 10000,
            "target_steps": 20000,
            "stopped_for_time": False,
            "latest_metrics": self._phase_b_metrics(),
            "specialization_checkpoints": [
                {
                    "enabled": True,
                    "checkpoint_step": 10000,
                    "label": "step_10000",
                    "records": 96,
                    "mi_bits": mi_bits,
                    "normalized_mi": 0.4,
                    "run_knockouts": run_knockouts,
                    "top_ablation_loss_deltas": (
                        [{"program": 3, "loss_delta": 0.08}] if run_knockouts else []
                    ),
                }
            ],
        }

    def _phase_b_knockout_report(self, *, mi_bits):
        return {
            "mutual_information": {"mi_bits": mi_bits, "normalized_mi": 0.4},
            "records": [{"category": "tool_choice"}],
            "ablations": [{"program": 3, "loss_delta": 0.08}],
            "specialization_metrics": {
                "knockout_selectivity": [{"program": 3, "selectivity_span": 0.1}]
            },
        }

    def _phase_c_specialization_report(self, *, program_order=(0, 1)):
        categories = ["tool_choice", "repair_after_failure"]
        role_vectors = {
            0: {
                "route": {"tool_choice": 0.92, "repair_after_failure": 0.08},
                "knockout": {"tool_choice": 0.12, "repair_after_failure": 0.01},
                "memory": [1.0, 0.0],
                "counts": [12, 0],
            },
            1: {
                "route": {"tool_choice": 0.08, "repair_after_failure": 0.92},
                "knockout": {"tool_choice": 0.01, "repair_after_failure": 0.12},
                "memory": [0.0, 1.0],
                "counts": [0, 12],
            },
        }
        selected_route = []
        knockout = []
        memory = []
        counts = [[0, 0], [0, 0]]
        for program, role in enumerate(program_order):
            selected_route.append(
                {
                    "program": program,
                    "preferred_category": (
                        "tool_choice" if role == 0 else "repair_after_failure"
                    ),
                    "selectivity_span": 0.84,
                    "by_category": role_vectors[role]["route"],
                }
            )
            knockout.append(
                {
                    "program": program,
                    "preferred_category": (
                        "tool_choice" if role == 0 else "repair_after_failure"
                    ),
                    "selectivity_span": 0.11,
                    "by_category": role_vectors[role]["knockout"],
                }
            )
            memory.append(
                {
                    "program": program,
                    "mean_vector": role_vectors[role]["memory"],
                    "mean_norm": 1.0,
                }
            )
            counts[0][program] = role_vectors[role]["counts"][0]
            counts[1][program] = role_vectors[role]["counts"][1]
        return {
            "checkpoint_step": 10000,
            "categories": categories,
            "mutual_information": {
                "mi_bits": 0.32,
                "normalized_mi": 0.64,
                "categories": categories,
                "programs": [0, 1],
                "counts": counts,
            },
            "specialization_metrics": {
                "selected_route_selectivity": selected_route,
                "knockout_selectivity": knockout,
            },
            "program_memory_summary": {
                "programs": memory,
                "mean_pairwise_cosine": 0.0,
            },
        }

    def _phase_d_rows(self):
        return [
            {
                "task_id": "multi_hop_chain_retrieval",
                "control_id": "tac_control_v1",
                "seed": 11,
                "primary_score": 0.72,
                "tokens_per_second": 400.0,
                "wall_clock_seconds": 10.0,
            },
            {
                "task_id": "multi_hop_chain_retrieval",
                "control_id": "parameter_matched_vanilla",
                "seed": 11,
                "primary_score": 0.61,
                "tokens_per_second": 3200.0,
                "wall_clock_seconds": 2.0,
            },
            {
                "task_id": "long_context_retrieval_4096",
                "control_id": "tac_control_v1",
                "seed": 11,
                "primary_score": 0.65,
                "tokens_per_second": 390.0,
                "wall_clock_seconds": 11.0,
            },
            {
                "task_id": "long_context_retrieval_4096",
                "control_id": "parameter_matched_vanilla",
                "seed": 11,
                "primary_score": 0.55,
                "tokens_per_second": 3300.0,
                "wall_clock_seconds": 2.0,
            },
            {
                "task_id": "tool_selection",
                "control_id": "tac_control_v1",
                "seed": 11,
                "primary_score": 0.79,
                "tokens_per_second": 420.0,
                "wall_clock_seconds": 9.0,
            },
            {
                "task_id": "tool_selection",
                "control_id": "parameter_matched_vanilla",
                "seed": 11,
                "primary_score": 0.68,
                "tokens_per_second": 3400.0,
                "wall_clock_seconds": 2.0,
            },
        ]


if __name__ == "__main__":
    unittest.main()
