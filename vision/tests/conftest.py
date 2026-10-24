import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_saved_roi_or_targets(tmp_path, monkeypatch):
    """Tests must not depend on the operator's saved mat box or clicked targets."""
    from scrub_vision import grasp, targets
    monkeypatch.setattr(grasp, "ROI_FILE", tmp_path / "no_roi.txt")
    monkeypatch.setattr(targets, "TARGETS_FILE", tmp_path / "no_targets.json")
