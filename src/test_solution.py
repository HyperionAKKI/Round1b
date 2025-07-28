# tests remain optional – this minimal smoke-test now uses new schema
import json
import os
import tempfile
from pathlib import Path

from src.main import DocumentIntelligenceSystem


def create_dummy_config() -> dict:
    return {
        "persona": {"role": "Test Analyst"},
        "job_to_be_done": {"task": "Find relevant sections"},
        "documents": [
            {"filename": "dummy.pdf", "title": "Dummy file"},
        ],
    }


def test_system():
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "input"
        in_dir.mkdir()
        out_dir = Path(tmp) / "output"

        # write dummy config
        cfg_path = in_dir / "config.json"
        cfg_path.write_text(json.dumps(create_dummy_config()))

        # touch an empty PDF so path exists
        (in_dir / "dummy.pdf").touch()

        # run
        DocumentIntelligenceSystem().run(in_dir, out_dir)

        assert (out_dir / "challenge1b_output.json").exists()


if __name__ == "__main__":
    test_system()
    print("✅ smoke-test passed")
