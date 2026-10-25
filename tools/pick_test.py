"""Scan -> detect -> for each requested part: pick with its jaw sizes, lift, photo, put back on the same spot.
usage: pick_test.py id [id ...]"""
import sys, time, json
from pathlib import Path
import numpy as np, cv2
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm"); sys.path.insert(0, R + "/perception"); sys.path.insert(0, R + "/perception/scripts")
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown
from scrub_arm.catalog import Resolved
from scrub_arm.handoff import TaskFailed
from scrub_vision.red_parts import RedPartsDetector, part_of
from fit_red_parts import LAYOUT
TRAY_ID = {"micro_scissors@side": "scissors_micro_back", "micro_scissors@flat": "scissors_micro_forward"}
out = Path(R + "/hw_check/real/steps"); out.mkdir(parents=True, exist_ok=True)
app = build(); d = app.driver; a = app.assistant; det = RedPartsDetector()
TARGET = None
want = [w for w in sys.argv[1:] if not w.startswith('--')]
import yaml
from scrub_vision.red_parts import Camera, red_mask, load_fingers
CAM = Camera(); AIM10 = yaml.safe_load(open(R + "/config/camera.yaml"))["aim_px"][10]


def centre_on_part(x, y, yaw, th, fid=None, tries=4):
    """At the 10 cm hover: put the fingertip spot (aim_px at 10 cm, measured by touching a dot) over the part's grab
    point seen in this photo -- along the part as well as across (a cut-off part: no correction)."""
    for _ in range(tries):
        time.sleep(0.4)
        for _ in range(6): cap.read()
        ok, img = cap.read(); q = d.q_measured()
        tip = CAM.px_to_mm([AIM10], q, th)[0]
        cands = [k for k in det.detect(img, q, grasp_offset=False) if fid in k.candidates and k.exposure == 1]
        if not cands:
            print("   centring: target not fully in view; no correction", flush=True); break
        k = min(cands, key=lambda k: np.hypot(k.grasp_mm[0] - x, k.grasp_mm[1] - y))
        e = np.array(k.grasp_mm) - tip
        print(f"   centring: off by ({e[0]:+.1f}, {e[1]:+.1f}) mm", flush=True)
        if np.linalg.norm(e) < 2.0: break
        a.move_line(0.6 * e[0], 0.6 * e[1], 0, speed=0.3)
        x, y = x + 0.6 * e[0], y + 0.6 * e[1]
    return x, y, yaw


a.refine = centre_on_part if '--centre' in sys.argv else None
_grasp = d.gripper_grasp
def grasp_and_look():
    ok = _grasp()
    snap(f"{CUR}_at_grasp"); print(f"   jaw stopped at {np.degrees(float(d.read_fresh()[-1])):.1f} deg, holding={ok}", flush=True)
    return ok
d.gripper_grasp = grasp_and_look
STEP = [0]
def stage(label):
    img = snap(f"{CUR}_{STEP[0]:02d}_{label}"); STEP[0] += 1
    if TARGET is None: return
    q = d.q_measured(); g = np.array(TARGET["grasp_mm"]); ax = np.radians(TARGET["yaw_deg"]); j = np.array([np.cos(ax), np.sin(ax)])
    for z in (0.0, 4.0):
        try:
            pts = det.cam.mm_to_px(np.array([g, g + 12 * j, g - 12 * j]), q, z).astype(int)
        except Exception:
            continue
        cv2.line(img, tuple(pts[1]), tuple(pts[2]), (255, 255, 0), 3); cv2.circle(img, tuple(pts[0]), 8, (255, 255, 0), -1)
    cv2.putText(img, label, (20, 50), 0, 1.4, (0, 255, 255), 3)
    cv2.imwrite(str(out / f"{CUR}_{STEP[0]-1:02d}_{label}.jpg"), img)
_width = d.gripper_width
def width_and_look(mm, wait_s=0.4):
    stage("hover_centred"); _width(mm, wait_s); stage("jaws_narrowed")
d.gripper_width = width_and_look
_line = a.move_line
def line_and_look(*args, **kw):
    r = _line(*args, **kw); stage("after_vertical_move"); return r
a.move_line = line_and_look
_grasp2 = d.gripper_grasp
def grasp_steps():
    stage("at_grasp_height_open"); ok = _grasp2(); stage("closed"); return ok
d.gripper_grasp = grasp_steps
d.start(); d.enable()
cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
def snap(name):
    time.sleep(0.5)
    for _ in range(6): cap.read()
    ok, f = cap.read(); cv2.imwrite(str(out / f"{name}.jpg"), f)
    json.dump({"q": d.q_measured().tolist(), "target": TARGET}, open(out / f"{name}.json", "w")); return f
try:
    for name in want:
        CUR = name
        d.move_pose("scan", speed=0.5); d.gripper_open()
        f = snap(f"{name}_scan"); q = d.q_measured()
        ref = np.array(LAYOUT[name] if name in LAYOUT else LAYOUT[[k for k in LAYOUT if TRAY_ID.get(k) == name][0]])
        tid = TRAY_ID.get(name, name)
        dets = [x for x in det.detect(f, q) if tid in x.candidates]
        if not dets:
            print(f"{name}: not seen", flush=True); continue
        exact = [x for x in dets if x.fine_id == tid or tid in x.alt_ids]
        x = min(exact or dets, key=lambda x: np.linalg.norm(np.array(x.grasp_mm) - ref))
        inst = app.tray.instruments[tid]
        th = det.instruments[tid].thickness_mm
        print(f"{name}: seen as {x.fine_id or x.candidates} at {np.round(x.grasp_mm, 1)} yaw {x.yaw_deg:.0f} "
              f"jaws {inst.jaw_open_mm} thickness {th}", flush=True)
        TARGET = {"grasp_mm": list(x.grasp_mm), "yaw_deg": x.yaw_deg, "axis_deg": x.axis_deg, "length_mm": x.length_mm,
                  "scan_q": q.tolist()}
        np.save(out / f"{name}_mask.npy", x.mask)
        try:
            a._pick_at(x.grasp_mm[0], x.grasp_mm[1], x.yaw_deg, th, fid=tid, jaw_open_mm=inst.jaw_open_mm)
        except TaskFailed as e:
            print(f"{name}: MISSED ({e})", flush=True); snap(f"{name}_missed"); continue
        a.holding = Resolved(slot=None, instrument=inst, orientation=None)
        g = float(d.read_fresh()[-1]); print(f"{name}: HOLDING, jaw at {np.degrees(g):.1f} deg", flush=True)
        snap(f"{name}_held")
        a._place_at(*a._picked_from)
        print(f"{name}: put back", flush=True)
    d.move_pose("scan", speed=0.5); d.gripper_open(); snap("after")
finally:
    cap.release(); park_and_shutdown(app)
