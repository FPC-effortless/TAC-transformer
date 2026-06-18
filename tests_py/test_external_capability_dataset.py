import unittest

from scripts.build_external_capability_dataset import (
    HfSource,
    convert_glaive,
    convert_gsm8k,
    convert_slimorca,
    convert_ultrachat,
    continuation_from_text,
)


def source(**overrides):
    values = {
        "key": "demo",
        "dataset": "demo/dataset",
        "config": "default",
        "train_split": "train",
        "eval_split": "test",
        "converter": convert_ultrachat,
        "train_cap_arg": "train_cap",
        "eval_cap_arg": "eval_cap",
        "stream": "assistant_qna",
        "license": "mit",
    }
    values.update(overrides)
    return HfSource(**values)


class ExternalCapabilityDatasetTests(unittest.TestCase):
    def test_ultrachat_uses_last_user_assistant_turn(self) -> None:
        row = {
            "__row_idx": 7,
            "messages": [
                {"role": "user", "content": "What is two plus two?"},
                {"role": "assistant", "content": "Four."},
                {"role": "user", "content": "Now answer in one word."},
                {"role": "assistant", "content": "Four"},
            ],
        }

        converted, reason = convert_ultrachat(row, source(), "train", 512)

        self.assertEqual(reason, "ok")
        assert converted is not None
        self.assertIn("Now answer", converted["prompt"])
        self.assertEqual(converted["answer"], "Four")
        self.assertEqual(converted["stream"], "assistant_qna")

    def test_slimorca_maps_human_gpt_roles(self) -> None:
        row = {
            "__row_idx": 3,
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Say hello."},
                {"from": "gpt", "value": "Hello."},
            ],
        }

        converted, reason = convert_slimorca(row, source(), "train", 512)

        self.assertEqual(reason, "ok")
        assert converted is not None
        self.assertIn("Say hello", converted["prompt"])
        self.assertEqual(converted["answer"], "Hello.")

    def test_chat_converter_trims_long_answer_to_complete_segment(self) -> None:
        row = {
            "__row_idx": 8,
            "messages": [
                {"role": "user", "content": "Explain briefly."},
                {
                    "role": "assistant",
                    "content": (
                        "First complete sentence. "
                        + "This sentence is intentionally repeated. " * 40
                    ),
                },
            ],
        }

        converted, reason = convert_ultrachat(row, source(), "train", 180)

        self.assertEqual(reason, "ok")
        assert converted is not None
        self.assertTrue(converted["answer"].endswith("."))
        self.assertLessEqual(
            len((converted["prompt"] + converted["answer"]).encode("utf-8")),
            180,
        )

    def test_gsm8k_extracts_final_answer_only(self) -> None:
        row = {
            "__row_idx": 1,
            "question": "Natalia sold 48 clips and then half as many. Total?",
            "answer": "She sold 24 more.\n#### 72",
        }

        converted, reason = convert_gsm8k(
            row,
            source(stream="private_reasoning_final_answer", converter=convert_gsm8k),
            "train",
            512,
        )

        self.assertEqual(reason, "ok")
        assert converted is not None
        self.assertEqual(converted["answer"], "72")
        self.assertNotIn("She sold", converted["answer"])

    def test_glaive_extracts_first_user_and_tool_call(self) -> None:
        row = {
            "__row_idx": 2,
            "system": 'SYSTEM: tools {"name": "get_weather", "parameters": {}}',
            "chat": (
                "USER: Weather in Lagos?\n\n"
                'ASSISTANT: <functioncall> {"name":"get_weather","arguments":"{\\"city\\":\\"Lagos\\"}"} '
                "<|endoftext|>\n\n"
                'FUNCTION RESPONSE: {"temp": 30}'
            ),
        }

        converted, reason = convert_glaive(
            row,
            source(stream="agentic_next_action", converter=convert_glaive),
            "train",
            512,
        )

        self.assertEqual(reason, "ok")
        assert converted is not None
        self.assertIn("get_weather", converted["prompt"])
        self.assertIn("<functioncall>", converted["answer"])

    def test_continuation_stays_within_context(self) -> None:
        text = " ".join(f"word{i}" for i in range(120))
        converted, reason = continuation_from_text(
            {"__row_idx": 4, "text": text},
            source(stream="english_lm_continuation"),
            "train",
            512,
            domain="english_lm_continuation:test",
        )

        self.assertEqual(reason, "ok")
        assert converted is not None
        self.assertLessEqual(
            len((converted["prompt"] + converted["answer"]).encode("utf-8")),
            512,
        )


if __name__ == "__main__":
    unittest.main()
