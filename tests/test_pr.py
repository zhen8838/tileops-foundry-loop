from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tileops_foundry_loop.pr import PRContractError, load_data, render_pr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "examples" / "fused-moe" / "pr-data.json"


class PRContractTests(unittest.TestCase):
    def setUp(self):
        self.data = load_data(DATA_PATH)

    def test_example_renders_uniform_title_and_public_sections(self):
        rendered = render_pr(self.data, DATA_PATH)
        self.assertEqual(
            rendered.title,
            "[Perf][foundry][MoE] Reduce sparse fused-MoE launch waste on H200",
        )
        self.assertLess(
            rendered.body.index("## Summary"),
            rendered.body.index("## TileFoundry Description"),
        )
        self.assertLess(
            rendered.body.index("## TileFoundry Description"),
            rendered.body.index("## Performance"),
        )
        self.assertIn("class RoutedExperts:", rendered.body)
        self.assertNotIn("import ", rendered.body)
        self.assertIn("| qwen3-235b-decode |", rendered.body)
        self.assertIn("| geometric mean |", rendered.body)

    def test_public_body_omits_private_evidence_and_paths(self):
        rendered = render_pr(self.data, DATA_PATH)
        for text in (
            "## Artifacts",
            "## Correctness",
            "## Reproduce",
            "authored_hir.py",
            "Entrypoint:",
            "Source:",
            "/home/",
            "/workspace/",
        ):
            self.assertNotIn(text, rendered.body)

    def test_example_has_all_comparators_on_all_workloads(self):
        rendered = render_pr(self.data, DATA_PATH)
        for comparator in self.data["comparators"]:
            self.assertIn(comparator["label"], rendered.body)
        for workload in self.data["workloads"]:
            self.assertIn(workload["label"], rendered.body)

    def test_missing_comparator_result_is_rejected(self):
        data = copy.deepcopy(self.data)
        del data["workloads"][0]["results"]["vllm"]
        with self.assertRaisesRegex(PRContractError, "every comparator"):
            render_pr(data, DATA_PATH)

    def test_invalid_scope_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["scope"] = "Fused MoE"
        with self.assertRaisesRegex(PRContractError, "scope must contain"):
            render_pr(data, DATA_PATH)

    def test_invalid_type_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["type"] = "Performance"
        with self.assertRaisesRegex(PRContractError, "type must be one of"):
            render_pr(data, DATA_PATH)

    def test_origin_cannot_be_reused_as_scope(self):
        data = copy.deepcopy(self.data)
        data["scope"] = "Foundry"
        with self.assertRaisesRegex(PRContractError, "reserved foundry origin"):
            render_pr(data, DATA_PATH)

    def test_non_positive_latency_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["workloads"][0]["results"]["candidate"]["median_ms"] = 0
        with self.assertRaisesRegex(PRContractError, "must be positive"):
            render_pr(data, DATA_PATH)

    def test_private_evidence_field_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["artifacts"] = [{"path": "/home/user/private.log"}]
        with self.assertRaisesRegex(PRContractError, "renderer-owned or private fields"):
            render_pr(data, DATA_PATH)

    def test_local_path_in_public_text_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["summary"][0] = "See /home/user/private.log for the result."
        with self.assertRaisesRegex(PRContractError, "local paths"):
            render_pr(data, DATA_PATH)

    def test_unknown_pr_field_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["notes"] = "A second contract does not belong here."
        with self.assertRaisesRegex(PRContractError, "unknown PR data fields"):
            render_pr(data, DATA_PATH)

    def test_description_import_is_rejected(self):
        data = copy.deepcopy(self.data)
        with tempfile.TemporaryDirectory() as temporary:
            data_path = Path(temporary) / "pr-data.json"
            data_path.write_text(json.dumps(data), encoding="utf-8")
            (data_path.parent / "authored_hir.py").write_text(
                "@module\nclass Kernel:\n    import os\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PRContractError, "must not contain imports"):
                render_pr(data, data_path)

    def test_hir_is_not_duplicated_in_pr_data(self):
        self.assertNotIn("tilefoundry_description", self.data)
        rendered = render_pr(self.data, DATA_PATH)
        self.assertIn("class RoutedExperts:", rendered.body)

    def test_template_is_valid_json(self):
        value = json.loads((ROOT / "templates" / "pr-data.json").read_text(encoding="utf-8"))
        self.assertEqual(value["comparators"][0]["role"], "candidate")
        self.assertNotIn("tilefoundry_description", value)


if __name__ == "__main__":
    unittest.main()
