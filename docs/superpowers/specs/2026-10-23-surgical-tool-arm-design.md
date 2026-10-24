# Instrument Identification + Pick Movement for the Scrub-Nurse Arm — Design

Date: 2026-10-23 (CalHacks). **Time budget: 30 hours to demo.**
Hardware: Seeed reBot Arm B601-RS (RobStride motors, CAN, driven from the M3 Max over a USB CAN adapter), wrist-mounted RGB webcam co-axial with the gripper and tilted toward it, Apple M3 Max 36 GB. No depth sensor. No tray: instruments lie **in a cluttered pile** on the table inside the arm's reach, touching and overlapping.

## 1. Scope

This document covers two systems and nothing else:

1. **Identification**: given a camera frame from a known arm pose, segment every visible 3D-printed instrument in the pile with a trained model, say which one it is, decide which ones are on top, and return a grasp pose in the arm's base frame.
2. **Movement**: given that grasp pose, pick the instrument with a contact-sensing descent (height in a pile is unknown), and run a **declutter loop**: if the requested instrument is buried, move the exposed ones to a staging area until it is on top.

ML is a hard requirement of the project. The segmentation model is the core of identification, not an add-on. Training data and policy testing run on **two parallel tracks**: a MuJoCo digital twin of the arm and pile (§5.8, primary) and a plain random-placement renderer (§5.2 step 1, backup). Whichever is producing good images first feeds the model; the backup never stops until the primary has proven itself on real photos. RL in simulation is the stretch (§5.6); a learned grasp-point predictor is the second stretch (§5.5).

**Out of scope, owned elsewhere** (the existing `arm/scrub_arm` package at github.com/Ishxn20/CalHacks and other teammates): voice capture and intent parsing (`catalog.py`), presenting the instrument forehand/backhand at the taught `handoff` pose, deciding when to release, the HTTP API, safety guard, teaching tool. This design plugs into those; it does not change them except where §6 says so.

Recognition must be **purely by appearance**: no tags, markers, colour bands or engraved-number reading.

## 2. Goals, non-goals, anti-goals

**Goals**
- Segment and identify every *exposed* instrument in a pile, including same-shape siblings that differ only by length.
- Return a grasp pose (x, y, yaw) accurate to about ±5 mm, ±5° in the plane; height found by contact.
- Retrieve a requested instrument that is buried by moving the instruments on top of it aside, one at a time.
- A trained model that can be shown: training curve, confusion matrix, live overlay.
- Everything runs on the M3 Max; no cloud at demo time.

**Non-goals**
- Identifying instruments that are mostly hidden (they get identified once uncovered).
- Grasping anything that is not roughly flat on top of the pile (the arm moves what's on top first).
- Multi-pose workspace scanning: the pile and the staging area fit in one camera view (my call; see §5.1).
- Blades as separate objects; collision planning beyond the pile; sutures as objects; engraved numbers; full hand-eye calibration.

**Anti-goals**
- Handing over a confidently wrong instrument. Moving a wrong instrument *aside* during decluttering is fine and expected.
- Crashing into the pile. Descent must stop on contact.
- A pipeline that only works under one lighting setup.
- Breaking the existing slot-based pick. It stays as the venue fallback.

## 3. What already exists (and the interfaces we must fit)

`arm/scrub_arm` is a working arm stack built around **a tray with taught slots**:

| Module | Provides | We use it as |
|---|---|---|
| `driver.py` | `move_to(q)` min-jerk joint moves, `gripper_open()`, `gripper_grasp()` → bool (did the jaws stop on something), `q_measured()`, `check_limits(q)`, collision guard | the only way we move the arm |
| `handoff.py` | job runner: `_pick(slot)` → `_verify` → `_present` → wait → release → home | we add `_pick_at(pose)` beside `_pick(slot)` |
| `verify.py` | arm POSTs `{expected, slot}` to a CV URL, expects `{ok, detected, confidence}` | our service answers it |
| `teach.py` / `config/poses.yaml` | teach and store named joint poses | used for `home`, `handoff`, `rest`, and our `scan_*` poses |
| `config/tray.yaml` (**repo root**, shared by arm, voice and CV as of commit d88b961) | instrument vocabulary + slot → poses | vocabulary kept; instrument ids are the shared key; slots become fallback |
| `voice/scrub_voice/listen.py` (new) | push-to-talk faster-whisper → `catalog.resolve` → writes `last_command.json` at repo root with `instrument` id | upstream of us; we consume the same instrument ids |
| `config/arm.yaml` | motor IDs/gains from Seeed's `rebotarm_rs.yaml`, joint limits from the RS URDF | unchanged |

**Catalog quirk to handle:** `catalog.resolve()` raises "there is no X on the tray" unless some slot in `tray.yaml` lists that instrument, and the voice layer calls it too. In pile mode availability comes from the camera, not slots. Smallest fix: when `pick_mode: ik`, `resolve` skips the slot filter and returns the instrument with `slot=None`; `_pick(slot)` is never called on that path. Every instrument in §4 must be listed under `instruments` (add `scissors_micro_forward`, `scissors_micro_back`, handle variants).

**Conflict to resolve:** the package has no inverse kinematics and cannot be given a target pose. "Anywhere in the workspace" needs one. The Seeed SDK (`reBotArm_control_py`, pip-installable, `pin>=3.9` has Apple-silicon wheels) ships `urdf/RS/urdf/ReBot_Arm_RS.urdf`, end-effector frame `gripper_end`, and `kinematics.inverse_kinematics.compute_ik(q_init, target_pos_m, roll=, pitch=, yaw=) -> IKResult(q, success, error, iterations)` plus `kinematics.forward_kinematics.compute_fk`. We call those; we write no kinematics.

Note: `compute_ik` loads the URDF via the SDK's `config/rebotarm.yaml` → `rebotarm_rs.yaml`. The wrapper in §6 passes the URDF path explicitly through `load_robot_model(urdf_path)` and `solve_ik` so it does not depend on the SDK's config files being present.

## 4. Instrument set and what separates each

STL bounding boxes (mm, longest × middle × thickness). Prints are 1:1 and **may be any colour, mixed within one set** (updated 2026-10-24: colour is never used as a cue; synthetic data randomises filament colour per instrument).

| Class | Length | Width | Thick | Separated by |
|---|---|---|---|---|
| Aortic cross-clamp (angled) | 250 | 79 | 4.5 | unique shape |
| Side-biting clamp (Satinsky) | 250 | 63 | 4.0 | unique shape (jaw geometry vs cross-clamp) |
| Castroviejo needle holder | 140 | 14 | 8 | unique: slim, no finger rings |
| Ryder needle holder | 180 | 51 | 3.0 | ring-handle family; length 180 |
| Mayo-Hegar short | 150 | 51 | 3.5 | ring-handle family; length 150 |
| Mayo-Hegar long | 200 | 51 | 3.8 | ring-handle family; length 200 |
| Potts scissors 45° | 180 | 47 | 3.0 | ring-handle family; angled blades |
| Micro scissors forward 45° | 150 | 13.6 | 9.6 | slim scissors; tip angle |
| Micro scissors back 135° | 150 | 13.6 | 12.3 | slim scissors; tip angle |
| DeBakey forceps | 200 | 13.8 | 9 | **same mesh scaled**; length 200 |
| Fine vascular forceps | 180 | 12.8 | 8 | **same mesh scaled**; length 180 |
| Resano forceps | 230 | 15.2 | 10 | **same mesh scaled**; length 230 |
| Scalpel #3 handle + #10/#11/#15 blade | 125+blade | 12 | 4.5 | handle family; length + blade silhouette |
| Scalpel #3L handle | 210+blade | 12 | 4.5 | handle family; length |
| Scalpel #4 handle + #20 blade | 135+blade | 13 | 4.5 | handle family; length + blade |
| Scalpel #7 slim handle | 160+blade | 11 | 4.5 | handle family; length, slimmer |

Facts the builder must design around:
- Three forceps and three handles are **the same mesh at different scales**. Shape cannot separate them; **metric length can**, and metric length is cheap from a calibrated camera at a fixed height.
- Blades (35–50 mm, 2 mm thick) are glued onto handles; each scalpel is one object. My call; the only sliced plate shows one of each handle and blade.
- Everything is flat (3–12 mm). Top-down pinch along the long axis is the only grasp.
- Prints are mirror-ambiguous when flipped. Matching considers both flips.
- The gripper region in the image (bottom ~25 %) is masked.
- Forward- and back-facing micro scissors have identical top-down silhouettes (the tip bend is out of the table plane); they are one visual class.
- Scalpel #10/#11/#15 share handle #3 and differ by 1–3 mm of blade: identification returns a candidate list, not a guess.
- Top-down reach (wrist vertical) is ~400 mm from the base at hover height: pile and staging must sit within ~350 mm.
- Coronary probes have no STL; out of scope.

## 5. Identification system

### 5.1 Scan pose and pixel → table mapping

The pile sits in a marked zone (~30×30 cm) with a clear staging zone beside it, both inside one camera view from a single taught `scan` pose (wrist straight down, ~40 cm up, taught with the existing `teach` tool). One view, one calibration, one homography. A second scan pose is only added if the workspace layout forces it.

Observed from a live frame: the webcam is wide-angle with visible barrel distortion, and the gripper fills roughly the bottom quarter of the image. So: (a) calibrate intrinsics once with a checkerboard (`cv2.calibrateCamera`) and undistort every frame before anything else; (b) the gripper mask is the bottom ~25 % of the frame, and the scan pose is taught so the pile sits in the upper two thirds of the view.

The homography maps undistorted pixels → (x, y) mm in the arm base frame on the **table plane**. Instruments on top of a pile sit 5–30 mm above the table, which shifts their apparent (x, y) by up to ~(height/400 mm) × distance-from-image-centre, i.e. a few millimetres. Acceptable for a pinch with 10 mm-wide jaws; the contact descent (§6.2) handles the height itself. **Touch calibration**, once per setup:
1. Scatter 6–8 tape dots across the workspace.
2. For each dot: jog the gripper tip onto it (`teach` → `limp`/`jog`), read `q_measured()`, run `compute_fk` → (x, y, z). Record; z is the table height.
3. For each scan pose: move there, grab a frame, click the visible dots, `cv2.findHomography(pixels, xy)`. Save to `config/camera.yaml` with the pose name and the gripper mask polygon.

The homography also gives mm/pixel for length measurement (slightly overestimated for raised instruments; the length gate tolerance absorbs it).

### 5.2 Core: instance segmentation model

Ultralytics YOLO11-seg (nano first, small if accuracy demands), fine-tuned to output one mask per visible instrument with a coarse class. Colour thresholding cannot do this job because touching green instruments merge into one blob.

**Classes (coarse, 10):** clamp_cross, clamp_satinsky, castroviejo, ring_needle_holder (Ryder + both Mayo-Hegars), potts, micro_scissors_fwd, micro_scissors_back, forceps (all three), scalpel (all handles). Fine identity within a family comes from the length gate (§5.3). Fewer classes → less data → trains in minutes and generalises better.

**Training data, in this order:**
1. **Synthetic piles** (the bulk, ~3,000 images, generated in the first hours), from two generators built in parallel:
   - **Primary, MuJoCo (§5.8):** drop 3–8 instruments into the pile zone and let physics settle them, then render top-down from the scan-pose camera. Piles have real resting angles, contacts and overlaps. Masks come from per-body segmentation rendering.
   - **Backup, trimesh/pyrender:** place 3–8 instruments at random positions, rotations and flips, **overlapping**, with an explicit z-order so occlusion is real. No physics; ready in ~2 h regardless of how MuJoCo setup goes.
   Both apply the same domain randomisation: green material with slight hue/brightness jitter, random table textures (white/grey sheets with wrinkles and shadows), camera at the scan-pose height ±10 %, mild blur and noise. Both write the same YOLO-seg label format so the training script does not care which produced an image. Train on the backup set first; swap in or mix MuJoCo images as soon as they exist.
2. **Real frames** (~150–300): photos of real piles from the scan pose. Label by running the model trained on synthetic data and correcting in a quick tool (Roboflow or Label Studio), or accept model labels where confidence is high. Fine-tune 1–2 more epochs with real data mixed in.
Train on the M3 Max MPS backend. Expect ~10–20 min for nano on 3k images.

**Occlusion / "is it on top" score:** for each mask, compare its outline to the class template (§5.3). A complete, unoccluded instrument matches well; a partly covered one matches poorly. `exposure = IoU_after_alignment`. Also count how many other masks' bounding boxes intersect it. Instruments with exposure above a threshold and the fewest intersections are "on top".

### 5.3 Fine identity, length gate, grasp point (classical, on top of the masks)

For each mask: PCA → principal axis; length in mm via the homography; **length gate**: fine classes within ±8 % of measured length (a 230 mm forceps is Resano, a 180 mm one is fine vascular). Then silhouette IoU against top-down templates rendered from the STLs (both mirror flips, both end orders) to settle the tip end and residual ambiguity. Each template carries a **grasp point** on the shank behind the jaws/blade (tip end, so the handle stays free) and a tip direction. Project the grasp point back through the homography → (x, y); yaw = axis angle in base frame. The length gate is only trusted when exposure is high; a partly covered instrument is reported as `identity: uncertain` and left for after decluttering.

**Grasp feasibility check:** at the chosen grasp point, the jaw footprint (jaw width × opening, rotated to yaw) must not intersect another mask that is *above* this one. If it does, slide the grasp point along the shank; if nothing works, this instrument is not graspable yet.

Known weaknesses: specular filament, deep shadows between instruments, a busy table, and instruments the same colour as the table. Mitigation: a plain mid-grey matte sheet (contrasts with both white and black prints), diffuse light from two sides.

### 5.4 Declutter policy (rule-based)

Given a requested instrument R and the scored masks:
1. If R is detected with high exposure and a feasible grasp → grasp R.
2. Else pick the instrument with the highest exposure and a feasible grasp (prefer ones whose mask overlaps R's region if R is partially visible) → grasp it, place it in the staging zone at a free spot, re-scan. Loop.
3. Stop after N = 6 moves or when nothing is graspable → `TaskFailed("I can't get to the <R>")`.
Staging placement: a grid of fixed (x, y) spots in the staging zone, filled left to right; instruments placed with their long axis along the grid row so they stay separated and remain pickable later.

### 5.5 Stretch model: learned grasp-point predictor

A small CNN (or a regression head on the YOLO backbone) that takes the crop of one segmented instrument and predicts (grasp u, v, yaw) and a success probability. Labels come for free: every real grasp attempt logs the crop, the grasp used, and `gripper_grasp()`'s True/False. Start from the template grasp points as pseudo-labels, fine-tune on logged outcomes. Supervised, ~1 h to build, improves visibly over the hackathon. Only if §5.2–5.4 are done and tested.

### 5.6 Stretch: RL grasp selection in simulation

With the MuJoCo twin in place (§5.8), a grasp-selection policy can be trained by trial and error in sim: state = segmented top-down image, action = which exposed instrument to pick and where on its shank, reward = success and progress toward exposing the target. Training is cheap because failures are free. **Transfer to the real arm is the risk**, especially RGB-only, so the policy is only switched on for the demo if it beats the rule-based declutter policy (§5.4) on real piles; otherwise it is shown as "trained in sim" with the rule policy driving. Learned world models (Dreamer-style) stay roadmap.

### 5.8 Simulation track: MuJoCo digital twin (primary data generator + policy test bed)

- Model: the B601-RS from Seeed's MuJoCo config (MJCF; otherwise convert `ReBot_Arm_RS.urdf` with MuJoCo's URDF loader), plus the 20 STLs as rigid bodies with approximate PLA mass (density ~1.24 g/cm³) and friction. Pile zone and staging zone as table markers at the real coordinates. Camera placed at the taught scan pose's FK, same intrinsics as the webcam (calibrated once with a checkerboard).
- Uses, in priority order:
  1. **Pile generator** for training data (§5.2). Physics-settled piles are closer to reality than random placement.
  2. **Declutter test bed**: run the §5.4 policy against hundreds of random piles overnight; count how often the target is reached within 6 moves and which failures recur. Fixes go into the rules.
  3. **RL training ground** (§5.6).
- Known limits: MuJoCo will not model a 3 mm print pinched off a soft sheet faithfully. Contact-sensing descent and the pinch itself are validated on hardware only (§6.5). The sim drives the real arm through the same joint interface, so the arm code is unchanged.
- Time: ~4 h to a first settled pile rendered with masks; ~2 h more for the declutter harness. If no rendered pile with correct masks exists by hour 8, the backup renderer carries the demo and the sim becomes a pitch-slide item.

### 5.7 Service interface

A separate process (`perception/`), owning the camera, HTTP on 127.0.0.1:8770:

- `POST /plan {"instrument": "<tray.yaml id>"}` → `{"action": "grasp_target" | "move_aside" | "fail", "target": {"id", "x_mm", "y_mm", "yaw_deg", "thickness_mm", "confidence", "exposure"}, "place": {"x_mm","y_mm","yaw_deg"} | null, "say": "..."}`. One call per declutter step; the arm executes it and calls again.
- `POST /verify {"expected": "<id>", "slot": ...}` → `{"ok", "detected", "confidence"}`. Existing contract from `verify.py`; answered by classifying the blob inside the gripper region while the arm holds the instrument at the taught `verify` pose.
- `GET /debug` → annotated JPEG of the last frame (masks, class, length, grasp point). This is the main debugging tool; build it first.

Instrument ids are the ones in `tray.yaml` `instruments`; add `scissors_micro_forward`, `scissors_micro_back`, and the handle variants there.

## 6. Movement system

### 6.1 Kinematics wrapper (`scrub_arm/kinematics.py`, new, ~60 lines)

- Load `ReBot_Arm_RS.urdf` once (vendored copy or the pip-installed SDK), frame `gripper_end`.
- `fk(q) -> (xyz_m, R)` via the SDK.
- `ik_topdown(x_mm, y_mm, z_mm, yaw_deg, q_seed) -> q | None`: target rotation = gripper pointing straight down (pitch = 180° about the base x axis, in the URDF's convention; **the builder verifies the sign on hardware with the FK of a taught wrist-down pose**), yaw = instrument axis. Call `solve_ik` seeded from `q_seed` (current joints); return None on `success == False`, error > 2 mm, or `driver.check_limits` failure. Try yaw and yaw+180° (a two-finger pinch is symmetric) and pick the solution closest to the seed.
- Unit check: for the taught `scan_1` pose, `ik_topdown(fk(q_scan))` must return q within 1° of `q_scan`.

### 6.2 `_pick_at` with contact-sensing descent (added to `handoff.py`, beside `_pick`)

Inputs: (x, y, yaw) from `/plan`. Height is unknown in a pile (table + 3 mm to table + 40 mm), so the descent **stops on contact**:
1. `HOVER`: z = table + 80 mm. IK fail → `TaskFailed("can't reach <instrument> there")`, no motion.
2. `gripper_open()`.
3. `DESCEND`: IK a short staircase of targets from hover down to z = table + thickness/2 − 2 mm (slight over-travel), stepping 3 mm, `move_to` each at `approach_speed_scale`. After each step read `driver.tracking_deviation()`; when any of joints 2–3 deviates more than `contact_deviation_deg` (new knob, ~1.5°, below the 25° collision guard) **or** the gripper's fingertip height per FK reaches the floor target, stop. Back off 1 mm. In MIT mode the arm is compliant, so a 1.5° lag means the fingertips are resting on something, not crushing it.
4. `gripper_grasp()`. False → lift to hover, call `/plan` again (the pile may have shifted), retry once; second miss → `TaskFailed`.
5. `LIFT`: back to hover z. Then either `_place_at(staging spot)` (declutter) or hand over to the existing `_verify` / `_present`.

`_place_at(x, y, yaw)`: hover → descend to table + thickness/2 + 5 mm (staging zone is flat; no contact sensing needed) → `gripper_open()` → hover.

### 6.3 Declutter loop (in the handoff job)

```
move_pose("scan")
for step in range(6):
    plan = POST /plan {instrument}
    if plan.action == "fail": raise TaskFailed(plan.say)
    _pick_at(plan.target)
    if plan.action == "grasp_target": break        # proceed to _verify / _present
    _place_at(plan.place); move_pose("scan")
else: raise TaskFailed("couldn't uncover it")
```
Every step is logged with the frame, masks, chosen grasp and `gripper_grasp()` result; that log is the training set for §5.5.

### 6.4 Fallback

`tray.yaml` slots and `_pick(slot)` stay. A config switch `pick_mode: ik | slots` in `arm.yaml` selects. If IK misbehaves at the venue, `slots` runs the demo with perception doing verification only.

### 6.5 Risks and the day-one tests

**Pinch of 3–4 mm flat parts on a hard table (bottom of the pile).** Test before anything else: lay a print on the planned surface, jog the gripper down, `close`. If it fails: 5 mm foam/silicone sheet, or print at 1.5× thickness, or a fingertip lip.

**Contact sensing threshold.** Test on day one too: jog down onto a print and watch `tracking_deviation()` to pick `contact_deviation_deg`. If MIT-mode feedback proves too noisy, fall back to a fixed descent to table + 25 mm with a wider jaw opening (works for the top layer only).

**Pile instruments shifting when a neighbour is lifted.** Handled by re-planning from a fresh frame after every move. Never reuse a plan.

**Reach.** Pile and staging zones are placed well inside the reachable area with the wrist vertical (build step 3 measures it).

**Gravity sag in MIT mode** (noted in the existing README) means FK of the commanded q differs from the real pose by a few degrees on joints 2–3. Touch calibration uses *measured* q, and `grasp_z_offset` absorbs the residual. If grasps are consistently off, `arm_mode: pos_vel` is the existing escape hatch.

## 7. Decided by the human partner

- Scope is identification + pick movement only; voice, presentation and release are owned elsewhere.
- Pure appearance recognition, no tags.
- Single wrist camera; no tray; instruments in a cluttered pile.
- ML is required; segmentation model is the core.
- MuJoCo simulation track runs in parallel with the plain renderer; the plain renderer is the backup.
- RL in sim is a stretch, enabled on the real arm only if it beats the rule policy.
- Hand-rolled motion on the SDK's IK, not MoveIt/ROS 2.
- Arm is driven from the M3 Max.

## 8. Left to the builder

- Thresholds (IoU, exposure, length tolerance, hover height, `contact_deviation_deg`), pile and staging zone positions, staging grid.
- Table surface (recommended: white or grey sheet, matte).
- Which handle+blade pairs get printed and therefore listed in `tray.yaml`.
- YOLO size (nano vs small), synthetic image count, how many real frames to hand-label.
- Whether perception imports `scrub_arm` in-process or stays a separate HTTP service (the latter matches the existing `verify` contract).

## 9. Build order (de-risking first)

1. Pinch test on the real prints and chosen surface (§6.5). Fix before anything else.
2. Teach `home`, `handoff`, `rest`, `verify`, `scan_1..n`; calibrate the gripper (`teach`). Kinematics wrapper; FK/IK round-trip check on `scan_1`; IK → `move_to` lands the gripper on three marked points within ±5 mm.
3. Touch calibration + per-pose homographies; `/debug` view; map the reachable area.
4. Tier 1 on the real prints at the scan poses; confusion matrix over the full set, both flips.
5. `_pick_at` end to end from `POST /handoff {"text": "..."}` with `pick_mode: ik`; confirm `pick_mode: slots` still passes the existing 39 tests.
6. `/verify` answered by perception.
7. Tier 2, auto-labelled from Tier 1.
8. Hardening: retries, failure strings, lighting test at the venue.
