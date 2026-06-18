import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments import benchmark_identity_compression_phase_boundary as bench


class IdentityCompressionPhaseBoundaryTests(unittest.TestCase):
    def test_routing_structure_metrics_report_program_utilization(self):
        output = SimpleNamespace(
            logits=torch.zeros(1, 2, 4),
            aux=SimpleNamespace(
                token_program_activations=torch.tensor(
                    [[[0.8, 0.2, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]]]
                ),
                token_selected_program_mask=torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]]
                ),
                selected_program_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            ),
        )

        metrics = bench.routing_structure_metrics(output)

        self.assertEqual(
            [round(value, 3) for value in metrics["route_program_utilization"].tolist()],
            [1.0, 0.5, 0.0, 0.0],
        )
        self.assertEqual(
            [
                round(value, 3)
                for value in metrics["activation_program_utilization"].tolist()
            ],
            [0.7, 0.3, 0.0, 0.0],
        )
        self.assertAlmostEqual(float(metrics["selected_programs_per_token"]), 1.5)
        self.assertAlmostEqual(float(metrics["route_zero_utilization_fraction"]), 0.5)
        self.assertGreater(float(metrics["route_top1_share"]), 0.60)
        self.assertLess(float(metrics["route_effective_programs"]), 4.0)

    def test_representational_thinning_metrics_report_state_and_energy_norms(self):
        output = SimpleNamespace(
            logits=torch.zeros(1, 2, 4),
            identity_states=[
                SimpleNamespace(
                    program_memory=torch.tensor(
                        [[[3.0, 4.0], [0.0, 2.0], [0.0, 0.0], [1.0, 1.0]]]
                    )
                )
            ],
            aux=SimpleNamespace(
                token_program_activations=torch.tensor(
                    [[[0.8, 0.2, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]]]
                ),
                token_selected_program_mask=torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]]
                ),
                selected_program_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            ),
        )
        energy_features = torch.tensor([[1.0, 2.0, 2.0, 1.0, 0.0, 0.0]])

        metrics = bench.representational_thinning_metrics(
            output,
            energy_features=energy_features,
        )

        self.assertAlmostEqual(float(metrics["selected_activation_mean"]), 0.6)
        self.assertAlmostEqual(float(metrics["selected_activation_l2"]), 0.6218253)
        self.assertAlmostEqual(float(metrics["identity_state_selected_norm"]), 3.5)
        self.assertGreater(float(metrics["identity_state_norm"]), 0.0)
        self.assertAlmostEqual(float(metrics["energy_feature_norm"]), 10.0**0.5)
        self.assertEqual(
            [round(value, 3) for value in metrics["identity_state_norm_by_program"].tolist()],
            [5.0, 2.0, 0.0, 1.414],
        )

    def test_estimator_finds_first_retention_drop_while_compression_improves(self):
        rows = [
            {
                "compression_strength": 0.00,
                "identity_retention_mean": 0.90,
                "compression_score": 0.20,
            },
            {
                "compression_strength": 0.02,
                "identity_retention_mean": 0.88,
                "compression_score": 0.30,
            },
            {
                "compression_strength": 0.05,
                "identity_retention_mean": 0.78,
                "compression_score": 0.42,
            },
            {
                "compression_strength": 0.10,
                "identity_retention_mean": 0.60,
                "compression_score": 0.50,
            },
        ]

        boundary = bench.estimate_identity_compression_boundary(
            rows,
            retention_drop_threshold=0.10,
        )

        self.assertEqual(boundary["boundary_status"], "crossed")
        self.assertAlmostEqual(boundary["boundary_strength"], 0.05)
        self.assertAlmostEqual(boundary["reference_retention"], 0.90)

    def test_aggregate_reports_required_metrics(self):
        rows = [
            {
                "compression_strength": 0.00,
                "seed": 7,
                "final_eval": {
                    "identity_retention_mean": 0.80,
                    "identity_retention_n0": 0.90,
                    "identity_retention_n5": 0.85,
                    "identity_retention_n10": 0.80,
                    "identity_retention_n20": 0.75,
                    "identity_retention_n50": 0.70,
                    "lm_accuracy": 0.70,
                    "energy_pair_accuracy": 0.90,
                    "rerank_accuracy": 0.80,
                    "routing_entropy": 0.60,
                    "activation_density": 0.50,
                    "active_program_fraction": 0.50,
                    "compression_score": 0.30,
                    "positive_compute_energy": 2.5,
                },
            },
            {
                "compression_strength": 0.05,
                "seed": 7,
                "final_eval": {
                    "identity_retention_mean": 0.65,
                    "identity_retention_n0": 0.80,
                    "identity_retention_n5": 0.70,
                    "identity_retention_n10": 0.65,
                    "identity_retention_n20": 0.60,
                    "identity_retention_n50": 0.50,
                    "lm_accuracy": 0.68,
                    "energy_pair_accuracy": 0.92,
                    "rerank_accuracy": 0.82,
                    "routing_entropy": 0.55,
                    "activation_density": 0.35,
                    "active_program_fraction": 0.45,
                    "compression_score": 0.45,
                    "positive_compute_energy": 2.4,
                },
            },
        ]

        result = bench.aggregate_phase_boundary_results(
            rows,
            distractor_counts=[0, 5, 10, 20, 50],
            retention_drop_threshold=0.10,
        )

        self.assertEqual(result["schema"], "identity_compression_phase_boundary.v1")
        self.assertEqual(result["phase_boundary"]["boundary_status"], "crossed")
        self.assertIn("identity_retention_n50", result["strength_summaries"][0])
        self.assertIn("routing_entropy", result["strength_summaries"][0])
        self.assertIn("critical_threshold_fit", result)

    def test_fitted_threshold_interpolates_between_grid_points(self):
        rows = []
        for seed, offset in [(7, 0.00), (19, 0.02)]:
            for strength, retention, density, compression in [
                (0.05, 0.60 + offset, 0.32, 0.44),
                (0.10, 0.56 + offset, 0.25, 0.47),
                (0.15, 0.48 + offset, 0.20, 0.49),
            ]:
                rows.append(
                    {
                        "compression_strength": strength,
                        "seed": seed,
                        "final_eval": {
                            "identity_retention_mean": retention,
                            "identity_retention_n0": retention,
                            "identity_retention_n5": retention,
                            "identity_retention_n10": retention,
                            "identity_retention_n20": retention,
                            "identity_retention_n50": retention,
                            "lm_accuracy": 0.70,
                            "energy_pair_accuracy": 0.90,
                            "rerank_accuracy": 0.80,
                            "routing_entropy": 0.60,
                            "activation_density": density,
                            "active_program_fraction": 0.50,
                            "compression_score": compression,
                            "positive_compute_energy": 2.5,
                        },
                    }
                )

        fit = bench.fit_critical_threshold_from_seed_rows(
            rows,
            distractor_counts=[0, 5, 10, 20, 50],
            retention_drop_threshold=0.10,
            bootstrap_samples=20,
        )

        self.assertEqual(fit["fit_status"], "crossed")
        self.assertAlmostEqual(fit["estimated_critical_strength"], 0.1375)
        self.assertAlmostEqual(fit["grid_crossing_strength"], 0.15)
        self.assertLess(fit["activation_density_slope"], 0.0)
        self.assertGreater(fit["compression_score_slope"], 0.0)
        self.assertIsNotNone(fit["estimated_strength_ci_low"])

    def test_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--compression-strengths",
                    "0.0",
                    "0.05",
                    "--seeds",
                    "3",
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--eval-batches",
                    "1",
                    "--eval-batch-size",
                    "1",
                    "--identity-trials",
                    "1",
                    "--distractor-counts",
                    "0",
                    "5",
                    "--seq-len",
                    "8",
                    "--vocab-size",
                    "32",
                    "--d-model",
                    "16",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "4",
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((output_dir / "identity_compression_phase_boundary.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
