import unittest

from scripts.prepare_unified_curriculum_for_tac import completion_row


class UnifiedCurriculumTacPrepTests(unittest.TestCase):
    def test_preference_completion_answer_is_assistant_content_only(self) -> None:
        row = {
            "id": "pref-1",
            "source": {"dataset": "unit"},
            "prompt": [{"role": "user", "content": "Return the code word."}],
            "chosen": [{"role": "assistant", "content": "ALPHA-17"}],
            "rejected": [{"role": "assistant", "content": "wrong"}],
        }

        prepared = completion_row("preference", row)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared["answer"], "ALPHA-17")
        self.assertNotIn("<|assistant|>", prepared["answer"])

    def test_sft_completion_uses_last_assistant_as_answer(self) -> None:
        row = {
            "id": "sft-1",
            "source": {"dataset": "unit"},
            "messages": [
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": "Return the code word."},
                {"role": "assistant", "content": "BRAVO-42"},
            ],
        }

        prepared = completion_row("sft", row)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared["answer"], "BRAVO-42")
        self.assertIn("<|user|>", prepared["prompt"])


if __name__ == "__main__":
    unittest.main()
