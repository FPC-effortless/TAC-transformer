import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_20m_from_scratch_dataset import (
    byte_chunks,
    stream_tokenized_memmap,
    write_local_seed,
)


class TwentyMillionFromScratchDatasetTests(unittest.TestCase):
    def test_byte_chunks_respect_max_and_min_bounds(self) -> None:
        text = "\n\n".join(" ".join([f"word{i}" for i in range(80)]) for _ in range(3))

        chunks = byte_chunks(text, chunk_bytes=300, min_bytes=120)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 300 for chunk in chunks))
        self.assertTrue(all(len(chunk.encode("utf-8")) >= 120 for chunk in chunks))

    def test_write_local_seed_serializes_existing_completion_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "seed.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "row1",
                        "stream": "assistant_qna",
                        "source": {"dataset": "demo"},
                        "text": (
                            "User: What is a compiler?\n"
                            "Assistant: A compiler translates source code into another form, "
                            "usually machine code, bytecode, or an intermediate representation. "
                            "It reads the program, checks structure, reports errors, and produces "
                            "an output that another system can execute or optimize."
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = tmp / "out.jsonl"
            with output.open("w", encoding="utf-8") as handle:
                stats = write_local_seed(
                    handle,
                    source,
                    split_kind="train",
                    target_bytes=10_000,
                    seen_hashes=set(),
                    rng=__import__("random").Random(1),
                )

            self.assertEqual(stats["written_records"], 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["stream"], "pretrain_seed")
            self.assertEqual(row["domain"], "pretrain_seed:assistant_qna")

    def test_stream_tokenized_memmap_writes_manifest_without_collecting_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "data.jsonl"
            source.write_text(
                json.dumps({"domain": "demo", "text": "hello world"}) + "\n"
                + json.dumps({"domain": "demo", "text": "second row"}) + "\n",
                encoding="utf-8",
            )

            manifest = stream_tokenized_memmap(
                source,
                tmp / "tokenized",
                vocab_size=512,
            )

            self.assertEqual(manifest["records"], 2)
            self.assertGreater(manifest["tokens"], len("hello world"))
            self.assertTrue(Path(manifest["tokens_path"]).exists())


if __name__ == "__main__":
    unittest.main()
