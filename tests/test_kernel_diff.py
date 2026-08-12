from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tileops_foundry_loop.kernel_diff import KernelDiffError, check_kernel_diff


class KernelDiffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", self.repo], check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.repo, check=True)
        self.path = self.repo / "src/tileops/kernels/example.py"
        self.path.parent.mkdir(parents=True)
        self.path.write_text(
            "@T.prim_func\ndef main(a):\n    a[0] = 1\n\ndef dispatch():\n    return 1\n",
            encoding="utf-8",
        )
        self.benchmark = self.repo / "benchmarks/ops/bench_example.py"
        self.benchmark.parent.mkdir(parents=True)
        self.benchmark.write_text("MODE = 'contract'\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=self.repo, check=True)
        self.base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()

    def tearDown(self):
        self.temporary.cleanup()

    def _commit(self, source: str) -> str:
        self.path.write_text(source, encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "change"], cwd=self.repo, check=True)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()

    def test_changed_prim_func_passes(self):
        head = self._commit(
            "@T.prim_func\ndef main(a):\n    a[0] = 2\n\ndef dispatch():\n    return 1\n"
        )
        check_kernel_diff(
            self.repo, self.base, head, "src/tileops/kernels/example.py:main"
        )

    def test_kernel_change_with_benchmark_change_fails(self):
        self.path.write_text(
            "@T.prim_func\ndef main(a):\n    a[0] = 2\n",
            encoding="utf-8",
        )
        self.benchmark.write_text("MODE = 'candidate'\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "change"], cwd=self.repo, check=True)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        with self.assertRaisesRegex(KernelDiffError, "benchmarks/ops/bench_example.py"):
            check_kernel_diff(
                self.repo, self.base, head, "src/tileops/kernels/example.py:main"
            )

    def test_kernel_change_with_kernel_correctness_test_passes(self):
        test_path = self.repo / "tests/kernels/test_example.py"
        test_path.parent.mkdir(parents=True)
        test_path.write_text("def test_kernel():\n    assert True\n", encoding="utf-8")
        head = self._commit("@T.prim_func\ndef main(a):\n    a[0] = 2\n")
        check_kernel_diff(
            self.repo, self.base, head, "src/tileops/kernels/example.py:main"
        )

    def test_dispatch_only_change_fails(self):
        head = self._commit(
            "@T.prim_func\ndef main(a):\n    a[0] = 1\n\ndef dispatch():\n    return 2\n"
        )
        with self.assertRaisesRegex(KernelDiffError, "unchanged"):
            check_kernel_diff(
                self.repo, self.base, head, "src/tileops/kernels/example.py:main"
            )

    def test_undecorated_symbol_fails(self):
        with self.assertRaisesRegex(KernelDiffError, "absent"):
            check_kernel_diff(
                self.repo, self.base, "HEAD", "src/tileops/kernels/example.py:dispatch"
            )


if __name__ == "__main__":
    unittest.main()
