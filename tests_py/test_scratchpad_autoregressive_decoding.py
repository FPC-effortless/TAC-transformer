import unittest

from experiments import benchmark_scratchpad_autoregressive_decoding as bench


class ScratchpadAutoregressiveDecodingTest(unittest.TestCase):
    def test_tac_decoder_learns_to_generate_from_verified_scratchpad_context(self):
        report = bench.run_scratchpad_autoregressive_decoding_probe(
            seed=3,
            train_steps=50,
        )

        self.assertEqual(
            report["decision"]["status"],
            "scratchpad_autoregressive_decoding_proved",
        )
        self.assertLess(
            report["training"]["final_loss"],
            report["training"]["initial_loss"],
        )
        self.assertGreaterEqual(report["scores"]["scratchpad_score"], 0.95)
        self.assertGreaterEqual(report["scores"]["counterfactual_score"], 0.95)
        self.assertLessEqual(report["scores"]["no_scratchpad_score"], 0.20)
        self.assertGreater(
            report["scores"]["scratchpad_score"],
            report["scores"]["no_scratchpad_score"],
        )
        self.assertTrue(report["sample_predictions"][0]["raw_completion"])


if __name__ == "__main__":
    unittest.main()
