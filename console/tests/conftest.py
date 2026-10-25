import re
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scrub_arm.app import build
from scrub_arm.audit import AuditLog
from scrub_arm.config import DEFAULT_TRAY_PATH
from scrub_arm.server import serve

ARM_CONFIG = Path(__file__).resolve().parents[2] / "arm" / "config"
TOKEN = "test-token"


@pytest.fixture
def arm_server(tmp_path):
    """A simulated arm (slot mode, sped up) behind the real HTTP server, with token auth and an audit log."""
    d = tmp_path / "config"
    shutil.copytree(ARM_CONFIG, d)
    shutil.copy(DEFAULT_TRAY_PATH, d)
    text = (d / "arm.yaml").read_text()
    text = re.sub(r"^(\s*mode:\s*)ik", r"\g<1>slots", text, flags=re.M)
    for key, val in {"open_deg": "0.0", "closed_deg": "-60.0", "calibrated": "false"}.items():
        text = re.sub(rf"^(\s*{key}:\s*)\S+", rf"\g<1>{val}", text, flags=re.M)
    (d / "arm.yaml").write_text(text)
    app = build(sim=True, config_dir=d)
    m = app.cfg.motion
    m.speed_scale, m.approach_speed_scale, m.min_move_s, m.settle_timeout_s = 20.0, 1.0, 0.02, 0.1
    app.cfg.gripper.settle_s = 0.15
    app.cfg.handoff.release_wait_s = 0.02
    app.cfg.handoff.present_timeout_s = 10.0
    app.cfg.control_rate_hz = 500
    app.driver.start()
    app.driver.enable()
    audit = AuditLog(tmp_path / "arm_audit.jsonl")
    httpd = serve(app.assistant, app.catalog, "127.0.0.1", 0, token=TOKEN, audit=audit)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{httpd.server_address[1]}", token=TOKEN, app=app, audit=audit)
    httpd.shutdown()
    httpd.server_close()
    app.driver.shutdown()


def wait_for(bridge, cond, timeout=10.0):
    """Poll the bridge until cond() holds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        bridge.poll()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"condition never held; arm status: {bridge.status.get('phase')} {bridge.status.get('last_job')}")
