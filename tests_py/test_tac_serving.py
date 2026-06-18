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

    def test_generate_tac_completion_can_use_shorter_context_window(self):
        model = _ScriptedByteModel([ord("O") + 4, ord("K") + 4, 3])

        result = generate_tac_completion(
            model,
            "abcdef",
            max_new_tokens=2,
            context_window=3,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            device="cpu",
        )

        self.assertEqual(result["completion"], "OK")
        self.assertEqual(result["context_window"], 3)
        self.assertEqual(result["checkpoint_max_seq_len"], 16)
        self.assertEqual(result["truncated_prompt_token_count"], 3)
        self.assertEqual(model.input_lengths, [3, 3])

    def test_generate_tac_completion_can_rerank_with_data_energy(self):
        low_energy_token = ord("B") + 4
        high_logit_token = ord("A") + 4
        model = _ScriptedEnergyModel(
            high_logit_token=high_logit_token,
            low_energy_token=low_energy_token,
        )

        result = generate_tac_completion(
            model,
            "prompt",
            max_new_tokens=1,
            temperature=0.0,
            top_k=2,
            top_p=1.0,
            energy_rerank_top_k=2,
            data_energy_weight=1.0,
            data_energy_verifier_threshold=5.0,
            device="cpu",
        )

        self.assertEqual(result["completion"], "B")
        self.assertEqual(result["generated_token_ids"], [low_energy_token])
        self.assertEqual(result["energy_rerank_top_k"], 2)
        self.assertEqual(result["data_energy_trace"], [0.0])
        self.assertEqual(result["data_energy_reranked_token_count"], 1)
        self.assertEqual(result["data_energy_verifier_required_count"], 0)
        self.assertGreaterEqual(model.candidate_score_calls, 2)

    def test_generate_tac_completion_verifier_can_replace_high_energy_token(self):
        replacement_token = ord("B") + 4
        selected_token = ord("A") + 4
        model = _ScriptedEnergyModel(
            high_logit_token=selected_token,
            low_energy_token=replacement_token,
            energy_by_token={selected_token: 10.0, replacement_token: 9.0},
        )
        verifier_calls = []

        def verifier(payload):
            verifier_calls.append(payload)
            self.assertEqual(payload["selected_token_id"], selected_token)
            self.assertEqual(payload["selected_data_energy"], 10.0)
            self.assertEqual(
                [candidate["token_id"] for candidate in payload["candidates"]],
                [selected_token, replacement_token],
            )
            return {
                "token_id": replacement_token,
                "reason": "candidate has verifier support",
            }

        result = generate_tac_completion(
            model,
            "prompt",
            max_new_tokens=1,
            temperature=0.0,
            top_k=2,
            top_p=1.0,
            energy_rerank_top_k=2,
            data_energy_weight=0.0,
            data_energy_verifier_threshold=5.0,
            data_energy_verifier=verifier,
            device="cpu",
        )

        self.assertEqual(result["completion"], "B")
        self.assertEqual(result["generated_token_ids"], [replacement_token])
        self.assertEqual(result["data_energy_verifier_required_count"], 1)
        self.assertEqual(result["data_energy_verifier_called_count"], 1)
        self.assertEqual(len(verifier_calls), 1)
        self.assertEqual(result["data_energy_verifier_actions"][0]["accepted"], True)
        self.assertEqual(
            result["data_energy_verifier_actions"][0]["replacement_token_id"],
            replacement_token,
        )

    def test_generate_tac_completion_rejects_context_window_above_checkpoint_limit(self):
        model = _ScriptedByteModel([ord("O") + 4])

        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            generate_tac_completion(
                model,
                "prompt",
                max_new_tokens=1,
                context_window=17,
                device="cpu",
            )

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
        self.input_lengths = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, **kwargs):
        self.input_lengths.append(input_ids.shape[1])
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 100.0
        return type("Output", (), {"logits": logits, "identity_states": []})()


class _ScriptedEnergyModel:
    def __init__(
        self,
        *,
        high_logit_token: int,
        low_energy_token: int,
        energy_by_token=None,
    ):
        self.config = TACConfig(
            vocab_size=260,
            d_model=8,
            n_heads=2,
            n_layers=1,
            n_programs=4,
            max_seq_len=16,
            program_embed_dim=4,
        )
        self.high_logit_token = int(high_logit_token)
        self.low_energy_token = int(low_energy_token)
        self.energy_by_token = (
            {int(token): float(energy) for token, energy in energy_by_token.items()}
            if energy_by_token is not None
            else None
        )
        self.candidate_score_calls = 0

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, **kwargs):
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[:, -1, self.high_logit_token] = 10.0
        logits[:, -1, self.low_energy_token] = 9.5
        if not kwargs.get("collect_auxiliary", False):
            return type("Output", (), {"logits": logits, "identity_states": []})()

        self.candidate_score_calls += 1
        last_token = int(input_ids[0, -1].detach().cpu())
        if self.energy_by_token is None:
            energy = 0.0 if last_token == self.low_energy_token else 10.0
        else:
            energy = self.energy_by_token.get(last_token, 10.0)
        aux = type(
            "Aux",
            (),
            {"data_energy": torch.full((1, input_ids.shape[1]), energy, device=input_ids.device)},
        )()
        return type(
            "Output",
            (),
            {"logits": logits, "identity_states": [], "aux": aux},
        )()


if __name__ == "__main__":
    unittest.main()
