"""Rehearsal rig: a simulated arm, the console and the dashboard in one process. No hardware, no camera.

The arm runs in slot mode (taught-pose picking from the demo tray in config/tray.yaml), so a handoff needs no
perception service. Everything on screen is labelled "Simulated arm". Use `--fast` to speed the arm up.
"""
from __future__ import annotations

import re
import secrets
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from scrub_arm.app import build
from scrub_arm.audit import AuditLog
from scrub_arm.config import DEFAULT_CONFIG_DIR, DEFAULT_TRAY_PATH
from scrub_arm.server import serve as serve_arm

from .app import build_console
from .config import Settings
from .server import serve as serve_console


def start(settings: Settings | None = None, fast: bool = False, agents: bool = False, port: int = 0,
          rehearse_droid: bool = True) -> SimpleNamespace:
    s = settings or Settings.from_env()
    work = Path(tempfile.mkdtemp(prefix="scrub-demo-"))
    config = work / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, config)
    shutil.copy(DEFAULT_TRAY_PATH, config)
    arm_yaml = config / "arm.yaml"
    arm_yaml.write_text(re.sub(r"^(\s*mode:\s*)ik", r"\g<1>slots", arm_yaml.read_text(), flags=re.M))

    app = build(sim=True, config_dir=config)
    if fast:
        m = app.cfg.motion
        m.speed_scale, m.approach_speed_scale, m.min_move_s, m.settle_timeout_s = 20.0, 1.0, 0.05, 0.2
        app.cfg.gripper.settle_s = 0.15
        app.cfg.handoff.release_wait_s = 0.05
    app.cfg.handoff.present_timeout_s = 300.0           # wait for the presenter to say "got it"
    app.driver.start()
    app.driver.enable()

    token = secrets.token_hex(16)
    arm_audit = AuditLog(work / "arm_audit.jsonl")
    arm_http = serve_arm(app.assistant, app.catalog, "127.0.0.1", 0, token=token, audit=arm_audit)
    threading.Thread(target=arm_http.serve_forever, name="sim-arm-http", daemon=True).start()

    s.arm_url, s.arm_token, s.audit_dir = f"http://127.0.0.1:{arm_http.server_address[1]}", token, work / "logs"
    console = build_console(s, sim=True, demo=True, rehearse_droid=rehearse_droid, arm_config=arm_yaml, agents=agents)
    stop = threading.Event()
    threading.Thread(target=console.bridge.run, args=(0.3, stop), name="poll", daemon=True).start()
    web = serve_console(console, "127.0.0.1", port)
    threading.Thread(target=web.serve_forever, name="console-http", daemon=True).start()

    def shutdown() -> None:
        stop.set()
        web.shutdown(); web.server_close()
        arm_http.shutdown(); arm_http.server_close()
        app.driver.shutdown()

    return SimpleNamespace(url=f"http://127.0.0.1:{web.server_address[1]}", console=console, app=app, work=work, stop=shutdown)
