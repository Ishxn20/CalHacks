# Real-arm runbook: camera picks + hand-off

How to set up, run, calibrate and recover the scrub-nurse arm with the wrist-camera tool recognition.
State as of 2026-10-25 (branch `cv-model-test`). Current calibration numbers: `docs/AGENT_HANDOFF.md` section 4.

---

## 1. What it does

`tools/pick.py`: you type a tool name, and the arm finds it on the white paper with the wrist camera, picks it up
and checks with the camera that it's off the table, retrying up to 3 times. Then it either hands it over (presents it
and lets go when the camera sees it has been taken) or puts it back.

The 8 parts (red 3D prints) it knows:

| You can type | Part | Grabbed at |
|---|---|---|
| `10 blade`, `#10` | #10 scalpel (~153 mm) | centre |
| `20 blade`, `twenty blade`, `#20` | #20 scalpel (~171 mm) | centre |
| `long scalpel` | long #11 scalpel (~236 mm) | centre |
| `tweezers`, `fine forceps` | fine vascular forceps (~187 mm) | centre |
| `forward micro scissors`, `micro scissors` | forward micro scissors (~156 mm) | thinnest point |
| `back micro scissors` | back-facing micro scissors (~156 mm) | thinnest point |
| `ryder` | Ryder needle holder (~185 mm) | thinnest point (neck) |
| `short needle holder` | short needle holder (~155 mm) | thinnest point (neck) |

The two micro scissors look identical from above when lying flat, so either request picks whichever one it sees
(user decision).

---

## 2. Hardware and one-time setup

- Arm: Seeed reBot B601-RS (6 joints + gripper, RobStride motors), CAN bus through a **PCAN-USB** adapter.
- Camera: USB wrist camera on the gripper, **OpenCV index 0**, 1920x1080.
- Work area: the parts lie on white paper, spread out (not touching), within about 380-560 mm in front of the base.
- Gripper fingers: the right finger is repaired with white tape; the finger mask was rebuilt for it.

Software (macOS):

1. **PCAN driver**: the MacCAN PCBUSB library, so `/usr/local/lib/libPCBUSB.dylib` exists.
2. **Python 3.12** venv at the repo root:
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -e "arm[hardware,kinematics,calibrate,dev]" -e "perception[dev]"
   ```
   Versions in use: numpy 2.5.4, opencv-python 5.0.0.93, pin (pinocchio) 4.1.0, motorbridge 0.5.6,
   ultralytics 8.4.185, torch 2.14.1, PyYAML 6.0.3.
3. **Camera permission**: run the scripts from a terminal app that macOS has allowed to use the camera
   (Terminal/iTerm; the in-app terminal panel worked). Without it, OpenCV prints `not authorized to capture video`
   and every frame is empty.

Every arm command below is run **from the repo root** with the PCAN library on the path:

```bash
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python <script> ...
```

(`DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib` also works.)

---

## 3. Everyday use

```bash
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/pick.py
```

```
tool> 20 blade
  try 1: #20 scalpel at (505, -104) mm, 16 mm wide there
  holding the #20 scalpel
  [h]and off or [p]ut back? h
  presenting: take the tool (or press Enter to release)
  handed over (the camera saw it taken); at rest
tool> q
```

- **h (hand off)**: the arm moves to the `handoff` pose. Pull the tool out of the jaws (they can't stop it). Half a
  second after the camera sees it gone, the jaws open and the arm goes to `rest`. **Enter** also releases. Not
  taken within 60 s: it puts the tool back.
- **p (put back)**: lowers it back where it came from and opens.
- Non-interactive: `tools/pick.py "10 blade" ryder` picks each one and puts it back.
- `q` or an empty line quits; the arm parks at `rest` and the motors are disabled.

**The jaws never open in mid-air while holding a tool.** They open only when it has been lowered to set it down, at
a hand-off, or on a camera-confirmed miss (the tool is still on the table, so the jaws are empty).

---

## 4. How it works (one pick)

1. **Scan**: the arm goes to the taught `scan` pose (camera looking straight down at the paper) and takes a photo.
2. **Find** (`perception/scrub_vision/red_parts.py`):
   - Red pixels: Lab a\* > 28, grown into connected a\* > 16 (pale ends). The wood and paper aren't red.
   - Each pixel goes to a metric top-down map of the table, through the arm's joints (forward kinematics) + the
     hand-eye calibration + the camera intrinsics, so sizes are in mm whatever the camera pose.
   - A part touching the frame edge or a finger (`perception/config/finger_mask.png`) counts as cut off: it gets
     candidates but no sure name.
   - Name: needle holders have ring holes; otherwise the nearest class in (length, mean width, median width,
     narrow-end width, taper, solidity), scaled by each class's spread (`perception/config/red_parts.yaml`, fitted on
     ~150 real wrist-camera views of two layouts, 600+ part samples). Held-out accuracy (first layout): 240/241 parts.
   - Grab point: the bounding-box centre (scalpels, forceps) or the thinnest cross-section (micro scissors, needle
     holders), plus `grasp_offset_mm`. The width there is measured too.
3. **Pick** (`arm/scrub_arm/handoff.py` `_pick_at`):
   - Hover 10 cm above the grab point with the scan pose's 45-degree gripper tilt (or turned 180 degrees if a joint
     would be near its limit).
   - Jaws open fully, then narrow to the tool's `jaw_open_mm` (`config/tray.yaml`) so they don't hit the neighbours.
   - One straight move down to grasp height: `table_z + tilt_tip_drop_mm + min(thickness/2, 1 mm)`.
   - Grasp: close softly until the jaws stall on the tool, then **4 degrees further** (`squeeze_deg`) for grip
     pressure. **Never past `closed_deg`** (a hard limit: going past it broke the gripper once).
   - Straight up.
4. **Check** (`tools/pick.py`): back to `scan` and look. If the tool is still lying within 20 mm of where it was, it
   missed: open, rescan, retry (up to 3).

---

## 5. Calibration values (and how to redo them)

| Value | Where | Now | Redo with |
|---|---|---|---|
| Camera intrinsics K, distortion | `config/camera.yaml` | rms 0.21 px | `python -m scrub_arm.calibrate charuco` |
| Hand-eye (camera on the gripper) | `config/camera.yaml` `handeye` | 9 views, 1.4 px (redone 2026-10-25 after the knock) | `python -m scrub_arm.calibrate handeye` |
| Gripper open / closed angles | `arm/config/arm.yaml` `open_deg` / `closed_deg` | **-0.31 / -209.58** (re-zeroed 2026-10-25; was 179.81 / -109.54) | teach tool: `g <deg>` until the tips just touch, then `gcal closed` |
| Jaw gap (mm) per gripper angle | `arm.yaml` `width_mm_at_deg` | 0-85 mm, ~0.33 mm/deg, still on the -215.03 closed zero: re-shift or re-measure | `tools/grip_sweep.py` (camera measures the fingers) |
| Fingertip drop when tilted | `arm.yaml` `pick.tilt_tip_drop_mm` | 7.0 | `tools/touch_table.py X Y YAW` (contact at a 9.2 mm tool-point height) |
| **Grab-point correction** | `perception/config/red_parts.yaml` `grasp_offset_mm` | **[-20.3, 2.5]** | `tools/grasp_calib.py X Y YAW` (see below) |
| Finger mask | `perception/config/finger_mask.png` | white-tape fingers, 14.4 % of the frame | `tools/capture_views.py OUT 12 24` then `tools/build_finger_mask.py OUT` |
| Scan pose joints | `red_parts.yaml` `scan_q` | measured at scan | `tools/snap_scan.py` saves the joints |
| Hand-off "taken" threshold | `tools/handoff_calib.json` | below 30 % of held (93k px held) | `tools/handoff_calib.py` |
| Per-tool open width | `config/tray.yaml` `jaw_open_mm` | 30-35 mm | width at the grab point + 20 mm |
| Parts classifier | `red_parts.yaml` `classes` | 9 classes x 6 features | `tools/refit_red_parts.py` on captured views |

### Grab-point correction (the one that drifts)

If picks start landing a consistent distance off the tools, the camera has probably moved on its mount. Re-measure:

1. Find an empty spot on the paper at least 30 mm from any part, e.g. around (487, -234) mm.
2. Run `tools/grasp_calib.py 487 -234 -85`. It does the real pick motion onto empty paper and photographs the
   fingertips open and closed (`hw_check/real/gcal/`).
3. Read the pixel where the closed fingertips meet. Convert it with `Camera().px_to_mm([[u, v]], q, 2)` (q from the
   saved json); `meet - target` is the error.
4. `grasp_offset_mm = -(meet - target)`, then check with `tools/pick_test.py` grasp photos and nudge by hand.

History: (-1.4, -10.6) from a 3 cm hover check (wrong at the table), then (-5.0, 4.1) from grasp height, then
(-20.3, 5.6) after the camera got knocked (the fingers sat ~125 px higher in the image), then hand-tuned from grasp
photos to (-20.3, 2.5) after the hand-eye was redone (2026-10-25).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PCAN initialize failed ... PCAN_ERROR_INITIALIZE` | Another program has the adapter open (another pick/teach/calibration still running: close it), or unplug and replug the adapter. |
| `motors not responding` / ping timeouts | Check with `hw_check/ping_each.py` (motors stay off). **Only joint 1 answers**: the CAN cable between joints 1 and 2 is loose, reseat it. **None answer**: adapter cable to joint 1, or motor power. |
| `FAULT: jointN is 25 deg off its setpoint ... frozen` right after a bus problem | Not a collision: that joint's feedback stopped updating. Fix the bus, then `hw_check/clear_faults.py`. |
| `disable_all failed: control ack timeout` on exit | A motor stopped answering mid-run; same bus checks. `hw_check/disable_each.py` shows which. |
| `the wrist camera gave no picture` / empty frames | Camera permission (see setup), or another program has camera 0 (e.g. `hw_check/live.py`). pick.py now retries and reopens the camera. Check indices with OpenCV: on this Mac, index 0 is the wrist camera. |
| Picks consistently a fixed distance off | Camera mount moved: re-measure the grab-point correction (section 5). |
| Jaws close but the tool slips out | Grab point on a wide or two-shank section (use thinnest point, see `GRAB_THINNEST` in red_parts.py), or `closed_deg` is not the real tip contact (recalibrate with `gcal closed`). |
| A part shows as `?name` / not found | It's cut off by the frame edge or a finger: move the paper further from the arm so every part is fully in the scan view. |
| Gripper strains or makes noise at closed | Never command past `closed_deg`; `set_gripper` clamps every command to [closed, open]. |

Read-only arm checks (nothing moves): `hw_check/ping_each.py`, `hw_check/disable_each.py`, `hw_check/clear_faults.py`.
Teach tool (manual jogging, poses, gripper calibration):
`DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python -m scrub_arm teach` from `arm/`; type `help`.

---

## 7. Files

| Path | What |
|---|---|
| `tools/pick.py` | type-a-tool pick, camera-checked retry, hand-off |
| `tools/README.md` | every real-arm script (scan snapshot, staged pick photos, view capture, refit, calibrations) |
| `perception/scrub_vision/red_parts.py` | the red-parts detector (segmentation, metric map, features, classes, grab points) |
| `perception/config/red_parts.yaml` | fitted classes, scan pose joints, grab-point correction |
| `perception/config/finger_mask.png` | where the fingers block the camera |
| `arm/scrub_arm/handoff.py` | pick / place / present motions |
| `arm/scrub_arm/driver.py` | gripper control (clamp, stall-then-squeeze) |
| `arm/config/arm.yaml`, `arm/config/poses.yaml` | arm limits, gripper calibration, pick settings, taught poses (`scan`, `handoff`, `rest`, `home`) |
| `config/camera.yaml` | camera intrinsics + hand-eye |
| `config/tray.yaml` | tool names, aliases, per-tool `jaw_open_mm` |
| `hw_check/real/` (gitignored) | photos + joints from every run: picks (`steps/`), calibrations (`gcal/`, `grip/`), views |

Tests: `.venv/bin/python -m pytest arm/tests perception/tests`. `test_lowering_is_one_straight_move_from_where_centring_left_it`
fails and has since before these changes.

---

## 8. Known limits

- The two micro scissors are one type: lying flat, they look the same from above.
- Parts must be spread out (not touching) and fully in the scan view.
- Hand-off detection needs the held tool in the camera's view at the `handoff` pose; if that pose is re-taught,
  rerun `tools/handoff_calib.py`.
- The grab-point correction is a single offset: re-measure it whenever the camera mount moves.
- Verified on the real arm after the latest recalibration: #20 scalpel and forceps (pick + camera hand-off). Earlier
  calibrations also picked the #10, long #11, Ryder and back micro scissors.
