from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryStructureTests(unittest.TestCase):
    def test_playbook_is_the_only_policy_document(self):
        self.assertTrue((ROOT / "PLAYBOOK.md").is_file())
        for legacy in (
            "docs",
            "prompts",
            "templates/tileops-pr.md",
            "templates/tilefoundry-gaps.md",
        ):
            self.assertFalse((ROOT / legacy).exists(), legacy)

    def test_agent_and_record_entrypoints_reference_playbook(self):
        for relative in (
            "AGENTS.md",
            "README.md",
            "plans/five-round.md",
            "templates/report.md",
            "templates/round-brief.md",
            "scripts/dispatch_round.sh",
            "scripts/send_review.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("PLAYBOOK.md", text, relative)

    def test_runner_image_default_is_not_duplicated_in_scripts(self):
        defaults = (ROOT / "config/defaults.env").read_text(encoding="utf-8")
        image_line = next(
            line for line in defaults.splitlines() if line.startswith("export TILEOPS_RUNNER_IMAGE=")
        )
        image = image_line.split("=", 1)[1]
        self.assertTrue(image)
        for script in (ROOT / "scripts").glob("*"):
            if script.is_file():
                self.assertNotIn(image, script.read_text(encoding="utf-8"), str(script))

    def test_public_example_config_has_no_machine_paths(self):
        text = (ROOT / "config/local.env.example").read_text(encoding="utf-8")
        for private_fragment in ("/home/", "/Users/", "qihang"):
            self.assertNotIn(private_fragment, text)


if __name__ == "__main__":
    unittest.main()
