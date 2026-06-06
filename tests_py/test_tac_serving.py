import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from tac_transformer.model import TACConfig, TACTransformerLM
from tac_transformer.serving import (
    decode_tac_byte_tokens,
    encode_tac_byte_tokens,
    generate_tac_completion,
    load_tac_checkpoint_for_generation,
    sample_next_token,
)
from scripts.prepare_tac_tokenized_corpus import build_arg_parser as build_tokenize_parser


class TACServingTests(unittest.TestCase):
    def test_byte_tokenizer_matches_tac_training_contract(self):
        token_ids = encode_tac_byte_tokens("A", vocab_size=260, append_eos=True)

        self.assertEqual(token_ids, [69, 3])
        self.assertEqual(decode_tac_byte_tokens(token_ids), "A")

    def test_sampling_controls_support_top_k_and_top_p(self):
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])

        self.assertEqual(sample_next_token(logits, temperature=0.0), 3)
        self.assertEqual(sample_next_token(logits, temperature=1.0, top_k=1), 3)
        sampled = sample_next_token(
            logits,
            temperature=1.0,
            top_p=0.6,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertIn(sampled, {2, 3})

    def test_generate_tac_completion_returns_sampling_metadata(self):
        model = _ScriptedByteModel([ord("O") + 4, ord("K") + 4, 3])

        result = generate_tac_completion(
            model,
            "prompt",
            max_new_tokens=8,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            device="cpu",
        )

        self.assertEqual(result["completion"], "OK")
        self.assertEqual(result["generated_token_count"], 2)
        self.assertEqual(result["temperature"], 0.0)
        self.assertEqual(result["top_k"], 1)
        self.assertEqual(result["top_p"], 1.0)
        self.assertEqual(result["tokenizer"], "tac_byte")

    def test_load_tac_checkpoint_for_generation_round_trips_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "tac.pt"
            config = TACConfig(
                vocab_size=260,
                d_model=8,
                n_heads=2,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "step": 12,
                    "best_eval_loss": 1.25,
                },
                checkpoint_path,
            )

            loaded, metadata = load_tac_checkpoint_for_generation(checkpoint_path)

            self.assertIsInstance(loaded, TACTransformerLM)
            self.assertEqual(metadata["model_type"], "tac")
            self.assertEqual(metadata["checkpoint_step"], 12)
            self.assertEqual(metadata["config"]["vocab_size"], 260)

    def test_tokenized_corpus_cli_parser_exposes_tac_defaults(self):
        parser = build_tokenize_parser()
        args = parser.parse_args(
            [
                "--train-jsonl",
                "train.prepared.jsonl",
                "--valid-jsonl",
                "eval.prepared.jsonl",
                "--output-dir",
                "tokenized",
            ]
        )

        self.assertEqual(args.vocab_size, 512)
        self.assertEqual(args.train_jsonl, Path("train.prepared.jsonl"))
        self.assertEqual(args.valid_jsonl, Path("eval.prepared.jsonl"))


class _ScriptedByteModel:
    def __init__(self, token_ids):
        self.config = TACConfig(
            vocab_size=260,
            d_model=8,
            n_heads=2,
            n_layers=1,
            n_programs=4,
            max_seq_len=16,
        )
        self.token_ids = list(token_ids)
        self.calls = 0

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, **kwargs):
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 100.0
        return type("Output", (), {"logits": logits, "identity_states": []})()


if __name__ == "__main__":
    unittest.main()
