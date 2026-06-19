import unittest


class FacadeImportTests(unittest.TestCase):
    def test_core_imports(self):
        from tac_transformer.core import TACConfig, TACTransformerLM
        self.assertIsNotNone(TACConfig)
        self.assertIsNotNone(TACTransformerLM)

    def test_memory_imports(self):
        from tac_transformer.memory import ProceduralMemoryStore, StructureMemoryModule
        self.assertIsNotNone(ProceduralMemoryStore)
        self.assertIsNotNone(StructureMemoryModule)

    def test_routing_imports(self):
        from tac_transformer.routing import LinearStructureBridge, SlotConditionedProgramBottleneck
        self.assertIsNotNone(LinearStructureBridge)
        self.assertIsNotNone(SlotConditionedProgramBottleneck)


if __name__ == "__main__":
    unittest.main()
