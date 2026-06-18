from __future__ import annotations

import unittest

from tac_transformer.procedural_memory import (
    ProceduralMemoryRead,
    ProceduralMemoryStore,
    ProceduralStep,
)


class TestProceduralMemoryStore(unittest.TestCase):
    def test_write_and_read_exact_task_key(self):
        store = ProceduralMemoryStore(max_records=4)
        record = store.write(
            task_key="unit-test",
            procedure_trace=[ProceduralStep(action="run failing test")],
            success_score=0.8,
        )
        read = store.read("unit-test")
        self.assertIsInstance(read, ProceduralMemoryRead)
        self.assertEqual(read.records[0].record_id, record.record_id)
        self.assertEqual(read.top_score, 0.8)

    def test_read_filters_by_min_success_score(self):
        store = ProceduralMemoryStore()
        store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="weak repair")],
            success_score=0.2,
        )
        read = store.read("task", min_success_score=0.5)
        self.assertEqual(read.records, [])
        self.assertEqual(read.top_score, 0.0)

    def test_read_returns_highest_scoring_records_first(self):
        store = ProceduralMemoryStore()
        low = store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="low")],
            success_score=0.3,
        )
        high = store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="high")],
            success_score=0.9,
        )
        read = store.read("task", top_k=2)
        self.assertEqual([record.record_id for record in read.records], [high.record_id, low.record_id])

    def test_update_success_tracks_running_average(self):
        store = ProceduralMemoryStore()
        record = store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="repair")],
            success_score=1.0,
        )
        store.update_success(record.record_id, success=False)
        self.assertLess(store.get(record.record_id).success_score, 1.0)
        store.update_success(record.record_id, success=True)
        self.assertGreater(store.get(record.record_id).success_score, 0.0)

    def test_store_evicts_lowest_value_records(self):
        store = ProceduralMemoryStore(max_records=2)
        store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="low")],
            success_score=0.1,
        )
        keep_a = store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="mid")],
            success_score=0.5,
        )
        keep_b = store.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="high")],
            success_score=0.9,
        )
        self.assertEqual({record.record_id for record in store.records}, {keep_a.record_id, keep_b.record_id})

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ProceduralMemoryStore(max_records=0)
        store = ProceduralMemoryStore()
        with self.assertRaises(ValueError):
            store.write(task_key="", procedure_trace=[ProceduralStep(action="x")], success_score=0.5)
        with self.assertRaises(ValueError):
            store.write(task_key="x", procedure_trace=[], success_score=0.5)
        with self.assertRaises(ValueError):
            store.read("x", top_k=0)


if __name__ == "__main__":
    unittest.main()
