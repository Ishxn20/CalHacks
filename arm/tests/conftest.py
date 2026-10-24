import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrub_arm.app import build  # noqa: E402
from scrub_arm.config import DEFAULT_TRAY_PATH  # noqa: E402


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    shutil.copy(DEFAULT_TRAY_PATH, d)
    return d


@pytest.fixture
def app(config_dir):
    """Simulated arm with motion sped up so a full handoff takes well under a second."""
    a = build(sim=True, config_dir=config_dir)
    m = a.cfg.motion
    m.speed_scale, m.approach_speed_scale, m.min_move_s, m.settle_timeout_s = 20.0, 1.0, 0.02, 0.1
    a.cfg.gripper.settle_s = 0.15
    a.cfg.handoff.release_wait_s = 0.02
    a.cfg.handoff.present_timeout_s = 5.0
    a.cfg.control_rate_hz = 500
    a.driver.start()
    a.driver.enable()
    yield a
    a.driver.shutdown()
