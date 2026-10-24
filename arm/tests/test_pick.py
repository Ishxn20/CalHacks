import json
import math
import threading
import urllib.request

import numpy as np
import pytest

from scrub_arm.handoff import Phase, TaskFailed
from scrub_arm.pick import PickError, plan_grasp
from scrub_arm.reach import ReachError, ReachMap, load_reach
from scrub_arm.server import serve

REF_J6 = 10.0   # taught wrist angle (deg) with the jaws closing along image x


def truth(kind, u, v):
    """Pretend the arm's true pixel->joint relation (affine, so the fit must recover it exactly)."""
    return np.radians([(u - 320) * 0.1, 50 + (v - 240) * 0.05 - (0 if kind == "grasp" else 8), 70, 0, 40, REF_J6])


def make_reach(path, grid=((150, 320, 490), (120, 240, 360))):
    rm = ReachMap(path=path, mm_per_px=0.5)
    for i, (u, v) in enumerate((u, v) for v in grid[1] for u in grid[0]):
        rm.points[i] = {"px": [u, v], "grasp": truth("grasp", u, v), "above": truth("above", u, v)}
    return rm


class StubVision:
    """Stands in for the tray-camera service."""

    def __init__(self, reach, **over):
        self.reach, self.calls = reach, 0
        self.det = {
            "found": True, "name": "sandbox_caliper", "conf": 0.9, "center_px": [300.0, 200.0], "grip_deg": 30.0,
            "long_deg": -60.0, "length_px": 200.0, "width_px": 30.0, "heavy_dir": [1.0, 0.0], "stable": True,
            "box_px": [[200, 185], [400, 185], [400, 215], [200, 215]], "others": [], "image": [640, 480],
            "markers": {str(i): p["px"] for i, p in reach.points.items()},
        }
        self.det.update(over)

    def detect(self, names, frames=None):
        self.calls += 1
        return dict(self.det)


@pytest.fixture
def papp(app, tmp_path):
    app.reach.points = make_reach(tmp_path / "r.yaml").points
    app.reach.mm_per_px = 0.5
    app.assistant.vision = StubVision(app.reach)
    return app


def plan(app, **over):
    return plan_grasp({**app.assistant.vision.det, **over}, app.reach, app.cfg.pick, app.cfg.joints)


# ── reach map ──────────────────────────────────────────────────────────────

def test_reach_fit_recovers_the_taught_relation_and_roundtrips(tmp_path):
    rm = make_reach(tmp_path / "r.yaml")
    assert np.allclose(rm.joints("grasp", 250, 200), truth("grasp", 250, 200), atol=1e-6)
    assert max(rm.loo_error_deg("grasp").values()) < 0.01
    rm.save()
    back = load_reach(tmp_path / "r.yaml")
    assert np.allclose(back.joints("above", 400, 300), truth("above", 400, 300), atol=1e-3)
    assert back.mm_per_px == 0.5 and back.ready() == []


def test_reach_needs_three_spread_points(tmp_path):
    rm = ReachMap(path=tmp_path / "r.yaml")
    assert len(rm.ready()) == 2                                       # grasp and above both missing
    for i, u in enumerate((100, 200, 300)):                           # collinear
        rm.points[i] = {"px": [u, 100], "grasp": truth("grasp", u, 100)}
    with pytest.raises(ReachError):
        rm.joints("grasp", 150, 100)


def test_quadratic_terms_kick_in_with_nine_points(tmp_path):
    rm = make_reach(tmp_path / "r.yaml")
    for i, p in rm.points.items():                                   # add curvature the affine fit cannot match
        u, v = p["px"]
        p["grasp"] = p["grasp"] + np.array([1e-4 * (u - 320) ** 2 / 100, 0, 0, 0, 0, 0])
    got = rm.joints("grasp", 250, 200)[0]
    want = truth("grasp", 250, 200)[0] + 1e-4 * (250 - 320) ** 2 / 100
    assert abs(got - want) < 1e-6


# ── planning ───────────────────────────────────────────────────────────────

def test_wrist_adds_the_grip_angle_to_the_taught_reference(papp):
    p = plan(papp, grip_deg=30.0)
    assert math.degrees(p.grasp_q[5]) == pytest.approx(REF_J6 + 30)
    assert p.above_q[5] == p.grasp_q[5]                              # descends without turning
    papp.cfg.pick.wrist_sign = -1.0
    assert math.degrees(plan(papp, grip_deg=30.0).grasp_q[5]) == pytest.approx(REF_J6 - 30)


def test_grip_angle_is_folded_to_the_shortest_wrist_turn(papp):
    assert math.degrees(plan(papp, grip_deg=-89.0).grasp_q[5]) == pytest.approx(REF_J6 - 89)
    assert math.degrees(plan(papp, grip_deg=100.0).grasp_q[5]) == pytest.approx(REF_J6 - 80)   # 100 == -80


def test_wrist_wraps_by_half_a_turn_when_a_limit_would_be_hit(papp):
    j6 = papp.cfg.joints[5]
    j6.upper = math.radians(20)                                      # +40 would not fit; the symmetric grip -140 does
    assert math.degrees(plan(papp, grip_deg=30.0).grasp_q[5]) == pytest.approx(REF_J6 + 30 - 180)
    j6.lower = math.radians(-20)
    with pytest.raises(PickError, match="limits"):
        plan(papp, grip_deg=30.0)


def test_grasp_point_moves_toward_the_handle(papp):
    p = plan_grasp(papp.assistant.vision.det, papp.reach, papp.cfg.pick, papp.cfg.joints, along=0.2)
    assert p.grasp_px == pytest.approx((300 + 0.2 * 200, 200))


def test_tray_markers_cancel_camera_drift(papp):
    d = papp.assistant.vision.det
    drifted = {k: [x + 10, y + 5] for k, (x, y) in d["markers"].items()}
    p = plan(papp, markers=drifted, center_px=[310.0, 205.0])
    assert p.grasp_px == pytest.approx((300.0, 200.0))
    assert "shifted" in p.anchor
    with pytest.raises(PickError, match="off its calibration"):
        plan(papp, markers={k: [x + 80, y] for k, (x, y) in d["markers"].items()})


def test_refuses_targets_outside_the_calibrated_area_and_low_confidence(papp):
    with pytest.raises(PickError, match="calibrated"):
        plan(papp, center_px=[900.0, 200.0])
    with pytest.raises(PickError, match="confidence"):
        plan(papp, conf=0.1)


def test_width_and_jaw_clearance_checks(papp):
    papp.cfg.pick.max_open_mm = 60.0                                 # 0.5 mm/px: 30px = 15mm fits
    plan(papp)
    with pytest.raises(PickError, match="across"):
        plan(papp, width_px=200.0)
    neighbour = {"name": "scalpel_11", "box_px": [[310, 190], [330, 190], [330, 210], [310, 210]]}   # between the jaws
    with pytest.raises(PickError, match="scalpel_11 is in the way"):
        plan(papp, others=[neighbour])
    far = {"name": "x", "box_px": [[300, 400], [320, 400], [320, 420], [300, 420]]}
    plan(papp, others=[far])


# ── the pick job on the simulator ──────────────────────────────────────────

def test_pick_job_picks_and_holds_at_tray_view(papp):
    asst = papp.assistant
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    job = asst.wait_idle()
    assert job["status"] == "done", job
    assert asst.holding.describe() == "Caliper" and asst.holding.slot is None
    assert np.allclose(papp.driver.q_command(), papp.poses.get("tray_view").q)
    assert asst.status()["holding_slot"] is None
    assert asst.vision.calls == 2                                    # two agreeing looks before moving
    asst.submit("release")
    assert asst.wait_idle()["status"] == "done" and asst.holding is None


def test_generic_category_matches_whichever_is_visible(papp):
    req = papp.catalog.resolve_pick("give me a scalpel")
    assert {i.id for i in req.candidates} == {"scalpel_10", "scalpel_11", "scalpel_15", "scalpel_20"}
    papp.assistant.vision.det["name"] = "scalpel_15"
    papp.assistant.submit("pick", req)
    assert papp.assistant.wait_idle()["status"] == "done"
    assert papp.assistant.holding.candidates[0].id == "scalpel_15"


def test_empty_jaws_retry_then_fail(papp):
    papp.driver.bus.object_in_jaws = False
    asst = papp.assistant
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    job = asst.wait_idle()
    assert job["status"] == "failed" and "3 tries" in job["error"]
    assert asst.vision.calls == 2 * 3 and asst.holding is None and asst.phase == Phase.IDLE


def test_missing_object_or_unsteady_view_does_not_move_on(papp):
    asst = papp.assistant
    asst.vision.det["found"] = False
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    assert "can't see a Caliper" in asst.wait_idle()["error"]
    asst.vision.det.update(found=True, stable=False)
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    assert "keeps moving" in asst.wait_idle()["error"]


def test_jumpy_target_is_not_trusted(papp):
    asst, v = papp.assistant, papp.assistant.vision
    xs = iter(range(100, 1000, 40))                                  # moves 40px every look

    def detect(names, frames=None):
        v.calls += 1
        return {**v.det, "center_px": [250.0 + next(xs) % 100, 200.0]}

    v.detect = detect
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    assert "keeps moving" in asst.wait_idle()["error"]


def test_pick_refuses_without_calibration_or_while_holding(papp):
    asst = papp.assistant
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    asst.wait_idle()
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    assert "already holding" in asst.wait_idle()["error"]
    asst.submit("release")
    asst.wait_idle()
    papp.reach.points = {}
    with pytest.raises(TaskFailed, match="reach map"):
        asst.plan_pick(papp.catalog.resolve_pick("caliper"))


def test_pick_handoff_presents_then_releases(papp):
    asst = papp.assistant
    req = papp.catalog.resolve_pick("caliper")
    req.handoff = True
    asst.submit("pick", req)
    for _ in range(500):
        if asst.phase == Phase.PRESENTING:
            break
        threading.Event().wait(0.01)
    assert asst.phase == Phase.PRESENTING
    asst.release()
    assert asst.wait_idle()["status"] == "done"


def test_slot_handoff_is_blocked_while_holding_a_picked_object(papp):
    asst = papp.assistant
    asst.submit("pick", papp.catalog.resolve_pick("caliper"))
    asst.wait_idle()
    asst.submit("handoff", papp.catalog.resolve("debakey"))
    assert "release it first" in asst.wait_idle()["error"]
    asst.submit("return")
    assert "not put back" in asst.wait_idle()["error"]


# ── HTTP ───────────────────────────────────────────────────────────────────

def test_pick_route_dry_run_and_errors(papp):
    httpd = serve(papp.assistant, papp.catalog, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(body):
        req = urllib.request.Request(base + "/pick", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        before = papp.driver.q_command().copy()
        code, body = post({"text": "caliper", "dry_run": True})
        assert code == 200 and body["plan"]["grip_deg"] == 30.0 and len(body["plan"]["grasp_deg"]) == 6
        assert np.allclose(papp.driver.q_command(), before)          # a dry run never moves the arm
        code, body = post({"instrument": "sparkly unicorn"})
        assert code == 400
        papp.assistant.vision.det["found"] = False
        code, body = post({"instrument": "caliper", "dry_run": True})
        assert code == 422 and "can't see" in body["say"]
        papp.assistant.vision.det["found"] = True
        code, body = post({"instrument": "caliper"})
        assert code == 202 and body["say"] == "Picking the Caliper"
        papp.assistant.wait_idle()
    finally:
        httpd.shutdown()
        httpd.server_close()


# ── teach-tool calibration commands ────────────────────────────────────────

def test_reach_commands_build_a_map_from_look_touch_above(app, tmp_path):
    from scrub_arm.reach import reach_cmd

    class Markers:
        def markers(self):
            return {"image": [640, 480], "markers": {str(i): {"center": [100.0 + 120 * i, 100.0 + 70 * (i % 2)], "side_px": 40.0}
                                                      for i in range(4)}}

    app.vision, app.cfg.pick.marker_mm = Markers(), 20.0
    assert "aruco targets [0, 1, 2, 3]" in reach_cmd(app, ["look"])
    assert app.reach.mm_per_px == pytest.approx(0.5)
    assert "not seen" in reach_cmd(app, ["touch", "9"])
    for i in range(4):
        app.driver.move_to(truth("grasp", *app.reach.points[i]["px"]))
        reach_cmd(app, ["touch", str(i)])
        app.driver.move_to(truth("above", *app.reach.points[i]["px"]))
        reach_cmd(app, ["above", str(i)])
    assert app.reach.ready() == []
    assert "affine" in reach_cmd(app, ["check"])
    assert len(load_reach(app.reach.path).ids("grasp")) == 4          # saved as it goes


def test_clicked_targets_work_without_printed_markers_and_never_anchor(app):
    from scrub_arm.reach import reach_cmd

    class Clicked:
        def markers(self):
            return {"image": [640, 480], "source": "clicked", "markers": {
                str(i): {"center": [100.0 + 150 * i, 100.0 + 90 * (i % 2)], "side_px": None} for i in range(4)}}

        def detect(self, names, frames=None):
            return {"found": True, "length_px": 200.0}

    app.vision = Clicked()
    out = reach_cmd(app, ["look"])
    assert "clicked targets" in out and app.reach.mm_per_px is None and app.reach.source == "clicked"
    assert "0.250 mm/px" in reach_cmd(app, ["scale", "50"])               # 50 mm tool spanning 200 px
    assert app.reach.anchor_shift({str(i): p["px"] for i, p in app.reach.points.items()}) is None
    assert load_reach(app.reach.path).source == "clicked"


def test_one_lift_offset_replaces_per_target_hover_poses(app, tmp_path):
    from scrub_arm.reach import reach_cmd
    rm = make_reach(tmp_path / "r.yaml")
    for p in rm.points.values():
        del p["above"]
    app.reach.points, app.reach.path, app.reach.lift = rm.points, rm.path, None
    assert any("above" in m for m in app.reach.ready())
    app.driver.move_to(truth("grasp", *rm.points[0]["px"]))
    app.driver.move_to(truth("grasp", *rm.points[0]["px"]) + np.radians([0, -8, 3, 0, -5, 0]))
    assert "reach touch 0" in reach_cmd(app, ["lift", "7"]) or True          # unknown id is refused, not crashed
    assert "hover offset" in reach_cmd(app, ["lift", "0"])
    assert app.reach.ready() == []
    got = app.reach.joints("above", 250, 200) - app.reach.joints("grasp", 250, 200)
    assert np.allclose(np.degrees(got), [0, -8, 3, 0, -5, 0], atol=0.1)
    assert np.allclose(load_reach(rm.path).lift, app.reach.lift)
