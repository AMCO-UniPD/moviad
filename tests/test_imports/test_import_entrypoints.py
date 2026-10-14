"""Check the documented training entrypoint without running a training job."""
from pathlib import Path
import subprocess
import sys


def test_import_entrypoints():
    import moviad.entrypoints
    import moviad.entrypoints.args

    example = Path(__file__).resolve().parents[2] / "examples/train_stfpm_mvtec.py"
    result = subprocess.run(
        [sys.executable, str(example), "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "--dataset-path" in result.stdout
