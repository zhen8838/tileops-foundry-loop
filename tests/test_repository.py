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

    def test_tilefoundry_runtime_delta_cannot_replace_runner_core(self):
        text = (ROOT / "config/tilefoundry-runtime-requirements.txt").read_text(
            encoding="utf-8"
        )
        for package in ("torch", "tilelang", "cuda", "vllm", "flashinfer"):
            self.assertNotIn(package, text.lower())
        self.assertIn("python-dateutil==", text)

    def test_preflight_checks_dependency_closure_and_real_tool_surfaces(self):
        preflight = (ROOT / "scripts/preflight.sh").read_text(encoding="utf-8")
        for script in (
            "check_tilefoundry_environment.py",
            "preflight_tilefoundry.py",
        ):
            self.assertIn(script, preflight)

    def test_foreman_integration_is_versioned(self):
        for relative in (
            "integrations/foreman/post-worktree.sh",
            "integrations/foreman/solo.md",
            "integrations/foreman/worker-env.sh",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        solo = (ROOT / "integrations/foreman/solo.md").read_text(encoding="utf-8")
        self.assertIn("`tileops-run`", solo)
        self.assertIn("tileops-run tilefoundry analyze", solo)

    def test_worker_wrapper_uses_sourced_path_not_local_install(self):
        env = (ROOT / "integrations/foreman/worker-env.sh").read_text(encoding="utf-8")
        wrapper = (ROOT / "scripts/tileops-run").read_text(encoding="utf-8")
        self.assertIn('export PATH="$TILEOPS_FOUNDRY_LOOP_ROOT/scripts:$PATH"', env)
        self.assertIn(
            "TILEOPS_WORKER_PYTHONPATH=/workspace/tileops-foundry-loop:/workspace/tileops",
            env,
        )
        self.assertIn('cd "$TILEOPS_ROUND_HOST"', env)
        self.assertIn("docker exec", wrapper)
        self.assertIn('--env "PYTHONPATH=$container_pythonpath"', wrapper)
        self.assertIn("with_gpu_lock.sh", wrapper)
        for unwanted in ("trun()", "tround()", "tf()", "gpu()", "tfgpu()"):
            self.assertNotIn(unwanted, env)
        self.assertNotIn(".local/bin", env)
        self.assertNotIn(".local/bin", wrapper)

    def test_dispatch_records_round_before_foreman_starts(self):
        dispatch = (ROOT / "scripts/dispatch_round.sh").read_text(encoding="utf-8")
        self.assertLess(
            dispatch.index("write_worker_admission.sh"),
            dispatch.index("foreman assign"),
        )


if __name__ == "__main__":
    unittest.main()
