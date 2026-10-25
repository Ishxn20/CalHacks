"""Voice text -> arm (sim, ik mode) -> real perception service over HTTP -> grasp -> present (or say why not)."""
import re
import threading

import numpy as np
import pytest

pytest.importorskip("scrub_vision")
from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh  # noqa: E402
from scrub_vision.service import VisionService, make_server  # noqa: E402
from scrub_vision.templates import load_instruments  # noqa: E402

from scrub_arm.app import build  # noqa: E402
from scrub_arm.handoff import Phase  # noqa: E402

CAM = CamSpec()
S = CAM.height_mm / CAM.fx
CENTER = (300.0, 0.0)          # camera looks down at this base-frame point from the scan pose
H = np.array([[S, 0, CENTER[0] - CAM.w / 2 * S], [0, S, CENTER[1] - CAM.h / 2 * S], [0, 0, 1]])


def scene(specs):
    insts = load_instruments()
    items = []
    for fid, x, y, yaw, level in specs:
        m = scene_mesh(insts[fid])
        items.append((m, pose(x - CENTER[0], y - CENTER[1], (12 if level else 0) + m.extents[2] / 2, yaw)))
    return render_scene(items, CAM, np.random.default_rng(0))


@pytest.fixture
def world(config_dir, tmp_path):
    clear = scene([("forceps_resano", 300, -20, 0.0, 0), ("needle_holder_ryder", 180, 110, 0.0, 0)])
    buried = scene([("forceps_resano", 300, -20, 0.0, 0), ("needle_holder_ryder", 340, -20, 1.4, 1)])
    current = {"img": clear, "clear": clear, "buried": buried}

    def source():
        return current["img"][0]

    class TrueInstances:
        """Stands in for the trained model: the renderer's per-instance masks (touching tools stay separate)."""

        def __call__(self, bgr, gripper_mask):
            return [(m & ~gripper_mask if gripper_mask is not None else m, None, 1.0) for m in current["img"][1]]

    gm = np.zeros(clear[0].shape[:2], bool)
    gm[int(clear[0].shape[0] * 0.85):] = True
    svc = VisionService(source, H, gm, TrueInstances(), log_dir=tmp_path / "logs")
    httpd = make_server(svc, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    p = config_dir / "arm.yaml"
    t = re.sub(r"^(\s*mode:\s*)slots", r"\g<1>ik", p.read_text(), flags=re.M)
    t = re.sub(r"^(\s*perception_url:\s*)\S+", rf"\g<1>http://127.0.0.1:{httpd.server_address[1]}", t, flags=re.M)
    p.write_text(t)
    app = build(sim=True, config_dir=config_dir)
    m = app.cfg.motion
    m.speed_scale, m.approach_speed_scale, m.min_move_s, m.settle_timeout_s = 20.0, 1.0, 0.02, 0.1
    app.cfg.gripper.settle_s = 0.05
    app.cfg.handoff.present_timeout_s = 5.0
    app.driver.start()
    app.driver.enable()
    yield app, svc, current
    app.driver.shutdown()
    httpd.shutdown()


def test_visible_resano_is_grasped_and_presented(world):
    app, svc, current = world
    app.assistant.submit("handoff", app.catalog.resolve("resano"))
    import time
    deadline = time.time() + 60
    while app.assistant.phase != Phase.PRESENTING and time.time() < deadline:
        assert app.assistant.status()["job"] is not None, app.assistant.status()["last_job"]
        time.sleep(0.05)
    assert app.assistant.phase == Phase.PRESENTING
    assert app.assistant.holding.instrument.id == "forceps_resano"
    plans = [e["msg"] for e in app.assistant.events if e["msg"].startswith("plan:")]
    assert plans and all(p.startswith("plan: grasp_target forceps_resano") for p in plans)
    app.assistant.release()
    assert app.assistant.wait_idle(timeout=30)["status"] == "done"


def test_covered_resano_is_left_alone_with_a_reason(world):
    """Nothing is moved out of the way (user decision): a covered target fails with speech, nothing picked."""
    app, svc, current = world
    current["img"] = current["buried"]
    app.assistant.submit("handoff", app.catalog.resolve("resano"))
    job = app.assistant.wait_idle(timeout=60)
    assert job["status"] == "failed" and "can't get to" in job["error"]
    assert app.assistant.holding is None and app.assistant._picked_from is None
