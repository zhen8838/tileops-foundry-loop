from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "template" / "check_round.py"
SPEC = importlib.util.spec_from_file_location("round_gate", CHECKER)
assert SPEC and SPEC.loader
round_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(round_gate)


HIR_A = """\
@module(entry="kernel", target="cuda:h200")
class Operator:
    @func
    def kernel(self, x):
        with Mesh(("cta",), (4,), ("block",)) as mesh:
            local = reshard(x, (128 @ mesh.block,), "smem")
            return reshard(local, (128 @ mesh.block,), "gmem")
"""

HIR_B = """\
@module(entry="kernel", target="cuda:h200")
class Operator:
    @func
    def kernel(self, x):
        with Mesh(("thread",), (128,), ("lane",)) as mesh:
            local = reshard(x, (4 @ mesh.lane,), "rmem")
            reduced = reduce_sum(local)
            return reshard(reduced, (4 @ mesh.lane,), "gmem")
"""

RUNTIME = """\
@runtime_module(Operator)
class Runtime:
    @runtime_func
    def kernel(self, x):
        return MainKernel(x)
"""


class RoundGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.round = self.root / "round"
        self.repo = self.root / "tileops"
        (self.round / "work").mkdir(parents=True)
        (self.round / "evidence").mkdir()
        subprocess.run(["git", "init", "-q", self.repo], check=True)
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=self.repo, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=self.repo,
            check=True,
        )
        kernel = self.repo / "src/tileops/kernels/example.py"
        kernel.parent.mkdir(parents=True)
        kernel.write_text(
            "@T.prim_func\ndef MainKernel(x):\n"
            "    for i in T.serial(16):\n        x[i] = 1\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=self.repo, check=True)
        self.base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        kernel.write_text(
            "@T.prim_func\ndef MainKernel(x):\n"
            '    for i in T.thread_binding(16, thread="threadIdx.x"):\n        x[i] = 1\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "schedule kernel"], cwd=self.repo, check=True
        )
        self._write_round()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_round(self) -> None:
        (self.round / "work/candidate_a.py").write_text(HIR_A, encoding="utf-8")
        (self.round / "work/final_hir.py").write_text(HIR_B, encoding="utf-8")
        (self.round / "work/runtime_twin.py").write_text(RUNTIME, encoding="utf-8")
        for name, value in (
            ("check.json", {"passed": True}),
            ("analyze-a.json", {"compute": "ok"}),
            ("analyze-b.json", {"compute": "ok"}),
            ("schedule-a.json", {"plan": "smem"}),
            ("schedule-b.json", {"plan": "rmem"}),
        ):
            (self.round / "evidence" / name).write_text(
                json.dumps(value), encoding="utf-8"
            )
        for name in ("correctness.log", "benchmark.log", "profile.log"):
            (self.round / "evidence" / name).write_text("passed\n", encoding="utf-8")
        (self.round / "report.md").write_text(
            "# Result\n\nMeasured.\n", encoding="utf-8"
        )
        (self.round / "findings.json").write_text(
            '{"findings": []}\n', encoding="utf-8"
        )
        (self.round / "pr-title.txt").write_text(
            "[Perf][foundry][Test] Schedule the kernel\n", encoding="utf-8"
        )
        (self.round / "pr-body.md").write_text(
            "## Summary\n\n- Structural schedule.\n\n"
            "## TileFoundry Description\n\n```python\n"
            + HIR_B
            + "```\n\n## Performance\n\nMeasured.\n\n"
            "## Result And Limitations\n\n**improvement without SOTA**\n",
            encoding="utf-8",
        )
        analysis = "tilefoundry analyze {hir} --compute-cost --memory --roofline --performance --json"
        provenance = {
            "tileops_base": self.base,
            "classification": "improvement without SOTA",
            "final_hir": "work/final_hir.py",
            "runtime_twin": "work/runtime_twin.py",
            "production_kernel": "src/tileops/kernels/example.py:MainKernel",
            "tilefoundry_check": {
                "command": "tilefoundry check work/runtime_twin.py:Runtime.kernel --json",
                "report": "evidence/check.json",
            },
            "iterations": [
                {
                    "name": "cta-smem",
                    "hir": "work/candidate_a.py",
                    "hypothesis": "stage by CTA",
                    "placement": "CTA split with smem staging",
                    "analysis": {
                        "command": analysis.format(hir="work/candidate_a.py:Operator"),
                        "report": "evidence/analyze-a.json",
                    },
                    "schedule": {
                        "command": "tilefoundry schedule work/candidate_a.py:Operator --topology cta --json --first-plan",
                        "report": "evidence/schedule-a.json",
                    },
                    "measured_ms": 2.0,
                    "verdict": "rejected",
                },
                {
                    "name": "thread-rmem",
                    "hir": "work/final_hir.py",
                    "hypothesis": "parallelize the scan",
                    "placement": "thread split with rmem reduction",
                    "analysis": {
                        "command": analysis.format(hir="work/final_hir.py:Operator"),
                        "report": "evidence/analyze-b.json",
                    },
                    "schedule": {
                        "command": "tilefoundry schedule work/final_hir.py:Operator --topology thread --json",
                        "report": "evidence/schedule-b.json",
                    },
                    "measured_ms": 1.0,
                    "verdict": "kept",
                },
            ],
            "decisions": [
                {
                    "analysis_fact": "rmem fits",
                    "schedule_fact": "threads overlap the scan",
                    "hir_change": "place the scan over threads",
                    "kernel_change": "thread-bound loop in MainKernel",
                }
            ],
            "correctness": {
                "command": "pytest test_example.py",
                "report": "evidence/correctness.log",
            },
            "benchmark": {
                "command": "pytest bench_example.py",
                "report": "evidence/benchmark.log",
            },
            "profile": {
                "command": "python profile.py",
                "report": "evidence/profile.log",
            },
        }
        (self.round / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

    def validate(self) -> dict:
        return round_gate.validate_round(self.round, self.repo)

    def test_complete_structural_round_passes(self) -> None:
        self.assertEqual(self.validate()["classification"], "improvement without SOTA")

    def test_hir_semantics_are_not_guessed_by_the_artifact_gate(self) -> None:
        (self.round / "work/final_hir.py").write_text(
            '@module(entry="kernel", target="cuda:h200")\n'
            "class Operator:\n    @func\n    def kernel(self, x):\n        return x\n",
            encoding="utf-8",
        )
        body = (self.round / "pr-body.md").read_text(encoding="utf-8")
        (self.round / "pr-body.md").write_text(
            body.replace(
                HIR_B,
                (self.round / "work/final_hir.py").read_text(encoding="utf-8"),
            ),
            encoding="utf-8",
        )
        self.assertEqual(self.validate()["classification"], "improvement without SOTA")

    def test_one_placement_fails(self) -> None:
        path = self.round / "provenance.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["iterations"] = data["iterations"][1:]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(round_gate.GateError, "at least two"):
            self.validate()

    def test_schedule_failure_is_not_gated(self) -> None:
        (self.round / "evidence/schedule-b.json").write_text(
            "PartitionSolveError: no feasible partition\n", encoding="utf-8"
        )

        self.assertEqual(self.validate()["classification"], "improvement without SOTA")

    def test_finding_classification_is_extensible(self) -> None:
        (self.round / "evidence/reproducer.log").write_text(
            "reproduced\n", encoding="utf-8"
        )
        (self.round / "findings.json").write_text(
            json.dumps(
                {
                    "findings": [
                        {
                            "id": "analysis-gap",
                            "classification": "analysis-limitation",
                            "command": "tilefoundry analyze work/final_hir.py:Operator",
                            "expected": "complete result",
                            "actual": "unsupported form",
                            "workaround_cost": "continued with measured evidence",
                            "public_surface": "tilefoundry analyze",
                            "affected_workloads": ["primary"],
                            "reproducer": "evidence/reproducer.log",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(self.validate()["classification"], "improvement without SOTA")

    def test_declared_kernel_file_must_change(self) -> None:
        subprocess.run(
            ["git", "reset", "--hard", self.base],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        with self.assertRaisesRegex(round_gate.GateError, "diff is empty"):
            self.validate()

    def test_description_must_match_final_hir(self) -> None:
        body = (self.round / "pr-body.md").read_text(encoding="utf-8")
        (self.round / "pr-body.md").write_text(
            body.replace(HIR_B, HIR_A), encoding="utf-8"
        )
        with self.assertRaisesRegex(round_gate.GateError, "exact final HIR"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
