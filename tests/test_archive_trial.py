from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "archive_trial.py"


class ArchiveTrialTests(unittest.TestCase):
    def test_redacts_machine_identity_and_omits_binary_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "archive"
            source.mkdir()
            (source / "report.md").write_text(
                "path=/home/person/work tmp=/tmp/build-user/cache.cc:7 "
                "workspace=/workspace/tileops root=/root/.cache mnt=/mnt/data "
                "cache=/ci-cache email=person@example.com   \r\n",
                encoding="utf-8",
            )
            (source / "tensor.pt").write_bytes(b"not public")

            subprocess.run(
                ["python", SCRIPT, source, destination], check=True, capture_output=True, text=True
            )

            report = (destination / "report.md").read_text(encoding="utf-8")
            self.assertEqual(
                report,
                "path=$HOST_HOME/work tmp=$TMP_PATH "
                "workspace=$CONTAINER_WORKSPACE/tileops root=$ROOT_HOME/.cache "
                "mnt=$MOUNT_ROOT/data cache=$CI_CACHE email=$REDACTED_EMAIL\n",
            )
            self.assertFalse((destination / "tensor.pt").exists())
            self.assertIn("tensor.pt", (destination / "MANIFEST.md").read_text(encoding="utf-8"))

    def test_rejects_credential_like_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "archive"
            source.mkdir()
            (source / "bad.log").write_text(
                "github_" + "pat_" + "abcdefghijklmnopqrstuvwxyz123456",
                encoding="utf-8",
            )

            process = subprocess.run(
                ["python", SCRIPT, source, destination], capture_output=True, text=True
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("credential-like content", process.stderr)


if __name__ == "__main__":
    unittest.main()
