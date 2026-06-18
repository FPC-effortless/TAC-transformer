import unittest

from scripts.build_capability_balanced_dataset import (
    format_action,
    parse_agentic_trajectory,
)


class CapabilityBalancedDatasetTests(unittest.TestCase):
    def test_parse_agentic_trajectory_extracts_goal_and_steps(self) -> None:
        text = """<record type="agentic_trajectory">
<system>
demo
<payload>
{"task":{"user_goal":"Fix tests.","environment":"py"},"sample_id":"x"}
<training_target>
[{"t":0,"action":{"tool":"run_shell","arguments":{"cmd":"pytest -q"}},"observation":{"status":"failed","exit_code":1}}]
</record>"""

        parsed = parse_agentic_trajectory(text)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        goal, steps = parsed
        self.assertEqual(goal, "Fix tests.")
        self.assertEqual(len(steps), 1)

    def test_format_action_keeps_tool_command_concise(self) -> None:
        action = {"tool": "read_file", "arguments": {"path": "src/core.py"}}

        self.assertEqual(format_action(action), "read_file: src/core.py")


if __name__ == "__main__":
    unittest.main()
