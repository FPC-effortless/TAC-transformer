import unittest

from experiments import benchmark_joint_tac_controller_training as bench


class JointTACControllerTrainingTest(unittest.TestCase):
    def test_tac_and_controller_train_together(self):
        report = bench.run_joint_tac_controller_training_probe(
            seed=5,
            train_steps=50,
        )

        self.assertEqual(
            report["decision"]["status"],
            "joint_tac_controller_training_proved",
        )
        self.assertLess(
            report["training"]["final_decoder_loss"],
            report["training"]["initial_decoder_loss"],
        )
        self.assertLess(
            report["training"]["final_controller_loss"],
            report["training"]["initial_controller_loss"],
        )
        self.assertGreater(report["parameter_updates"]["tac_max_abs_delta"], 0.0)
        self.assertGreater(report["parameter_updates"]["controller_max_abs_delta"], 0.0)
        self.assertGreater(report["gradient_flow"]["tac_policy_grad_abs_sum"], 0.0)
        self.assertGreaterEqual(report["scores"]["scratchpad_score"], 0.95)
        self.assertGreaterEqual(report["scores"]["controller_scratchpad_score"], 0.95)
        self.assertGreaterEqual(report["scores"]["controller_simulation_score"], 0.95)
        self.assertGreaterEqual(report["scores"]["controller_teaching_score"], 0.95)
        self.assertLessEqual(report["scores"]["no_scratchpad_score"], 0.20)


if __name__ == "__main__":
    unittest.main()
