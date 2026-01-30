import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_cli_help_module():
    # Ensure the module entrypoint runs and prints help
    proc = subprocess.run(
        [sys.executable, "-m", "swimrs.cli", "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "SWIM-RS workflow CLI" in proc.stdout or "usage:" in proc.stdout.lower()
