import json
import threading
import urllib.request

import numpy as np
import pytest

from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh
from scrub_vision.service import VisionService, make_server
from scrub_vision.templates import load_instruments

CAM = CamSpec()
S = CAM.height_mm / CAM.fx
H = np.array([[S, 0, -CAM.w / 2 * S], [0, S, -CAM.h / 2 * S], [0, 0, 1]])
STAGING = [(250.0, -200.0), (250.0, -260.0)]


def frame(specs):
    """Rendered frame plus its exact per-instance masks."""
    insts = load_instruments()
    items = [(scene_mesh(insts[f]), pose(x, y, scene_mesh(insts[f]).extents[2] / 2, yaw)) for f, x, y, yaw in specs]
    return render_scene(items, CAM, np.random.default_rng(0))


class TrueInstances:
    """Stands in for the trained model: these tests are about the service, not segmentation quality."""

    def __init__(self, svc_holder):
        self.h = svc_holder

    def __call__(self, bgr, gripper_mask):
        return [(m & ~gripper_mask if gripper_mask is not None else m, None, 1.0) for m in self.h["masks"]]


@pytest.fixture
def server(tmp_path):
    world = {}
    img, world["masks"] = frame([("forceps_resano", -60, -60, 0.0), ("scissors_potts", 120, 40, 1.0)])
    gm = np.zeros(img.shape[:2], bool)
    gm[int(img.shape[0] * 0.8):] = True
    svc = VisionService(lambda: img, H, gm, TrueInstances(world), STAGING, log_dir=tmp_path)
    svc.world = world
    httpd = make_server(svc, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", svc, tmp_path
    httpd.shutdown()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_plan_from_injected_frame(server):
    url, _, logs = server
    p = post(url + "/plan", {"instrument": "forceps_resano"})
    assert p["action"] == "grasp_target" and p["target"]["id"] == "forceps_resano"
    assert abs(p["target"]["x_mm"] - (-14)) < 15   # tip at x=+55, grasp 0.3*230 mm back from it
    assert list(logs.glob("*.json")) and list(logs.glob("*.jpg"))


def test_plan_unknown_instrument_is_fail_not_500(server):
    url, _, _ = server
    p = post(url + "/plan", {"instrument": "probe_coronary"})
    assert p["action"] == "fail"


def test_plan_marks_staging_used(server):
    url, svc, _ = server
    img, svc.world["masks"] = frame([("forceps_resano", 0, -20, 0.0)])  # target only
    svc.frame_source = lambda: img
    post(url + "/plan", {"instrument": "scissors_potts"})                 # not visible -> move resano aside
    assert svc.used_staging == {0}
    post(url + "/reset_staging", {})
    assert svc.used_staging == set()


def test_verify_matches(server):
    url, svc, _ = server
    img, svc.world["masks"] = frame([("forceps_resano", 0, 60, np.pi / 2)])   # reaching into the gripper zone
    svc.frame_source = lambda: img
    v = post(url + "/verify", {"expected": "forceps_resano", "slot": None})
    assert v["ok"] is True and v["detected"] == "forceps_resano"
    v = post(url + "/verify", {"expected": "scissors_potts", "slot": None})
    assert v["ok"] is False


def test_debug_returns_jpeg(server):
    url, _, _ = server
    post(url + "/plan", {"instrument": "forceps_resano"})
    with urllib.request.urlopen(url + "/debug", timeout=10) as r:
        assert r.headers["Content-Type"] == "image/jpeg" and r.read(2) == b"\xff\xd8"


def test_detections_off_the_work_mat_are_dropped(tmp_path):
    """Only instruments mostly on the mat count (wood grain off the mat produced false detections)."""
    from scrub_vision.service import region_mask

    world = {}
    img, world["masks"] = frame([("forceps_resano", -60, -60, 0.0), ("scissors_potts", 120, 40, 1.0)])
    gm = np.zeros(img.shape[:2], bool)
    mat_mm = [(-200, -150), (40, -150), (40, 50), (-200, 50)]          # covers resano, not potts
    svc = VisionService(lambda: img, H, gm, TrueInstances(world), STAGING, log_dir=tmp_path,
                        region=region_mask(mat_mm, H, img.shape[:2]))
    ids = {d.fine_id for d in svc.detect(img)}
    assert "forceps_resano" in ids and "scissors_potts" not in ids
    no_mat = VisionService(lambda: img, H, gm, TrueInstances(world), STAGING, log_dir=tmp_path)
    assert len(no_mat.detect(img)) == 2
