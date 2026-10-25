# Agent handoff: scrub-nurse arm (state 2026-10-25)

For the next agent working on the **voice / service side**: getting spoken requests to move the real arm. It covers
what runs today, every calibration number, the HTTP APIs, and the gaps between the service path and the script that
actually works on the real arm. All paths are relative to the repo root,
`/Users/aditya/Documents/Programming/Hackathon/CalHacks`.

Setup, recovery and troubleshooting steps live in [`docs/REAL_ARM_RUNBOOK.md`](REAL_ARM_RUNBOOK.md); this file has
everything else.

---

## 1. Branches: read this first

| Branch | What's on it |
|---|---|
| `cv-model-test` (this branch) | The arm, the camera tool recognition and every calibration below. Last code commit `2aabd10b`. |
| `main` (`581a345b`) | All of `cv-model-test` merged in, **plus** the teammates' work that is not on this branch: `console/` (the voice console: Claude intent, Cartesia STT/TTS, arm client, dashboard), `gesture/`, arm HTTP API hardening (bearer token, Origin/Host checks, audit log), perception `POST /count`, 20 Hz arm readings. |

The voice/service code you'll most likely work on (`console/`) is **only on `main`**. Section 6 describes both.
Merging was done with a normal merge (`581a345b`); never force-push `main`: it has teammates' commits.

---

## 2. What works today

- **`tools/pick.py` is the one path proven on the real arm.** You type a tool name; it scans with the wrist camera,
  picks the tool, confirms by camera that it left the table (retries up to 3), then hands it over (presents it and
  opens when the camera sees it taken, or Enter) or puts it back.
- Picks verified 2026-10-25 after recalibration, for the three tools in current use: **#10 scalpel, #20 scalpel,
  fine vascular forceps ("tweezers")**. Last measured run: 5 of 6 first try, 6/6 within retries. The other five parts
  have not been re-verified since the camera was knocked.
- The camera identifies all 8 printed parts. The two micro scissors are deliberately one shared type (identical
  from above when flat; user decision).
- **The HTTP service path (`scrub_arm serve` + perception service + voice) has not been run on the real arm.** See
  section 7 for what it lacks compared with `tools/pick.py`.

```bash
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/pick.py              # interactive: tool> 10 blade
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/pick.py "20 blade"   # pick it, then put it back
```

---

## 3. Hardware

| Item | Detail |
|---|---|
| Arm | Seeed reBot B601-RS, 6 joints + gripper, RobStride motors (joints 1-3 `rs-06`, 4-6 and gripper `rs-00`), motor ids 0x01-0x07, feedback id 0xFD |
| CAN | PCAN-USB adapter, macOS MacCAN PCBUSB library at `/usr/local/lib/libPCBUSB.dylib`; channel `can0` maps to `PCAN_USBBUS1`. Every arm command needs `DYLD_LIBRARY_PATH=/usr/local/lib`. Only **one process** can hold the bus. |
| Camera | USB wrist camera on the gripper, OpenCV index 0, 1920x1080. Only one process can hold it. macOS needs camera permission for the terminal app (the Claude terminal panel works; a sandboxed shell gets empty frames). |
| Gripper | Tapered fingers; the right one is wrapped in white tape (repair). The tips flex round a part, so a grasp stalls ~3 deg from closed even on a 13 mm scalpel: the gripper angle **can't** tell whether a tool is held; the camera check does. |
| Work area | Parts on white paper, spread out, about x 380-560 mm in front of the base. Paper corners (`mat_mm`): (523, 121), (552, -225), (295, -178), (304, 138) mm. |

---

## 4. Calibration numbers (all current values)

### Arm (`arm/config/arm.yaml`)

| Key | Value | Notes |
|---|---|---|
| `arm_mode` | `mit` | control rate 200 Hz, `active_report: true` |
| Joint limits (deg) | j1 [-160, 160], j2 [0, 179], j3 [0, 179], j4 [-89, 89], j5 [-89, 89], j6 [-180, 180] | max vel 90/60/60/120/120/180 deg/s |
| MIT gains (kp, kd) | j1 50/3, j2 150/10, j3 150/10, j4 50/5, j5 50/4, j6 50/4 | |
| `gripper.open_deg` / `closed_deg` | **-0.31 / -209.58** | **`closed_deg` is a hard limit**: `set_gripper` clamps every command; going past it broke the gripper once. See the warning below. |
| `gripper.width_mm_at_deg` | (-215.0, 0.0), (-185.9, 9.7), (-157.0, 18.5), (-127.9, 28.5), (-98.9, 37.7), (-70.0, 47.5), (-41.2, 57.5), (-12.2, 67.3), (17.0, 77.0), (42.0, 85.3) | deg -> jaw gap mm, ~0.33 mm/deg |
| `gripper.move_kp/kd`, `grip_kp/kd` | 20 / 1, 6 / 0.5 | |
| `gripper.settle_s`, `squeeze_deg`, `squeeze_kp` | 0.6 s, 4.0 deg, 20 | grasp = close softly until the jaws stall, then 4 deg tighter |
| `gripper.holding_min_gap_deg` | 2.5 | unreliable (see Hardware) |
| `motion.speed_scale`, `approach_speed_scale` | 0.5, 0.25 | |
| `motion.tracking_error_limit_deg` | 25 | collision guard: "joint N 25 deg off" also fires when a joint's CAN feedback stops |
| `pick.mode` | `ik` | camera-located picks (not taught slots) |
| `pick.perception_url` | `http://127.0.0.1:8770` | |
| `pick.table_z_mm`, `hover_mm` | 0.0, 100.0 | |
| `pick.tilt_pose` | `scan` | grasps copy the scan pose's 45 deg gripper tilt |
| `pick.tilt_tip_drop_mm` | 7.0 | tilted fingertips sit ~9 mm below the tool point |
| `handoff.orientation_offsets_deg` | forehand 0, backhand 180 (joint 6) | default forehand |
| `handoff.present_timeout_s`, `release_wait_s` | 60, 0.5 | |
| `handoff.tug.enabled` | `false` | measured: a pull deflects the wrist too little to detect; the camera does it in `pick.py` |
| `verify.url` | `null` | verification disabled |
| `server.host` / `port` | 127.0.0.1 / 8765 | |

> **Gripper calibration warning.** `open_deg`/`closed_deg` were re-recorded with `gcal` (-0.31 / -209.58) *after*
> the width table was shifted onto the previous zero (closed -215.03, open 41.99). The table still says 0 mm at
> -215.0, which is now past `closed_deg`. Until the table is re-shifted or re-measured, millimetre jaw widths are
> off by ~5 deg (~2 mm) near closed and the open span differs (209 deg now vs 257 before). Fix: confirm the jaws'
> real closed/open with `teach` (`s`, `gcal`), then shift the table by `new closed - (-215.03)` or re-measure it with
> `tools/grip_sweep.py`. History: closed -109.54 / open 179.81 (before 2026-10-25) -> -215.03 / 41.99 -> -209.58 / -0.31.

### Poses (`arm/config/poses.yaml`, joints deg j1..j6, all `taught: true`)

| Pose | Joints |
|---|---|
| `rest` | 3.12, 0.9, 0.05, 0.24, -0.16, -2.98 |
| `home` | 3.12, 40.9, 40.05, -29.76, -0.16, -2.98 |
| `scan` | 3.11, 90.11, 110.05, -64.77, -0.15, -3.13 (camera looks straight down at the paper) |
| `handoff` | 3.11, 120.11, 130.05, 20.23, -0.15, -90.0 |

`staged`, `verify` and the slot poses (`A1.above` ...) are placeholders (`taught: false`); the real arm refuses them.

### Camera (`config/camera.yaml`)

| Key | Value |
|---|---|
| Intrinsics `K` | fx 1087.18, fy 1085.74, cx 927.87, cy 473.83 (1920x1080), reprojection 0.21 px |
| `dist` | [-0.03841, 0.10961, 0.00169, 0.00099, -0.08331] |
| `handeye.p_tool_cam_m` | **[-0.0712, -0.00074, 0.08554]** |
| `handeye.R_tool_cam` | [[-0.002694, -0.519082, 0.85472], [-0.999976, -0.004106, -0.005645], [0.00644, -0.854714, -0.519059]] |
| `handeye` fit | 9 views, **1.41 px**, board squares 20.78 mm, camera moved 117 mm across views (redone 2026-10-25; views in `hw_check/handeye2/`, the first 3 of 12 dropped because the board moved) |
| Check | with this hand-eye, the parts measure their true lengths with no correction: #10 153 mm, #20 171, forceps 187 |
| `table_z_mm` | 2.12 |
| Previous hand-eye (before the knock) | p [-0.08573, 0.01258, 0.07394], 7 views, 1.67 px; kept in `perception/tests/test_red_parts.py` for the old test frame |

### Vision (`perception/config/red_parts.yaml`, `perception/scrub_vision/red_parts.py`)

| Item | Value |
|---|---|
| `grasp_offset_mm` | **[-20.3, 2.5]** added to every grab point (table x, y mm; +y is left in the scan image). Tuned by hand on 2026-10-25 from grasp photos; re-measure with `tools/grasp_calib.py` if picks land a fixed distance off. |
| `scan_q` (rad) | [0.06213, 1.60996, 1.88494, -1.15857, -0.00268, -0.05177] |
| Segmentation | Lab a* hysteresis: seeds > 28, grown > 16 |
| Canvas | metric top-down, 0.5 mm/px over table x 150-750, y -350-350 mm, parts projected at 4 mm above the table |
| Cut-off rule | a part within 12 px of the frame edge or a finger counts as cut off (`UNSEEN_MARGIN_PX`); cut off at one end, its grab point is half its class length in from the visible end |
| Classifier | 6 features [length, area/length, median width, narrow-end width, taper, solidity]; ring holes >= 2 -> needle holder; reject > 4 sigma, ambiguous if runner-up within 0.5 sigma |
| Grab point | oriented-box centre; thinnest cross-section for `micro_scissors`, `needle_holder_short`, `needle_holder_ryder` |
| Finger mask | `perception/config/finger_mask.png`, rebuilt 2026-10-25 from 24 views (`tools/build_finger_mask.py`), 14.4 % of the frame; old one kept as `finger_mask_prev.png` (untracked) |

Class means (length, area/L, median width, narrow width mm, taper, solidity):

| Class | Mean | Notes |
|---|---|---|
| `scalpel_10` | 153.1, 11.6, 12.6, 10.1, 0.8, 1.0 | |
| `scalpel_20` | 170.7, 13.7, 14.8, 11.8, 0.8, 1.0 | |
| `scalpel_11_long` | 235.7, 12.7, 13.5, 11.8, 0.9, 1.0 | |
| `forceps_fine_vascular` | 187.2, 7.9, 9.0, 4.7, 0.5, 0.9 | |
| `micro_scissors@flat` / `@side` / `@open` | 156.3 / 156.4 / 148.7 long | one part, three resting modes |
| `needle_holder_ryder` | 185.1, 11.0, 9.0, 6.1, 0.7, 0.5 | ringed |
| `needle_holder_short` | 154.8, 12.5, 10.6, 7.0, 0.7, 0.7 | ringed |

### Tools and hand-off (`config/tray.yaml`, `tools/handoff_calib.json`, `tools/pick.py`)

| Item | Value |
|---|---|
| `jaw_open_mm` (open width coming down) | #10 33, #20 35, long #11 34, forceps 30, micro scissors 31, Ryder 30, short needle holder 30 (= width at the grab point + 20 mm; measured grab widths #20 15, #10 13, forceps 10 mm) |
| `in_set` (what camera picks may take) | scalpel_10, scalpel_20, scalpel_11_long, forceps_fine_vascular, scissors_micro_forward, scissors_micro_back, needle_holder_short, needle_holder_ryder |
| Aliases | "10 blade", "20 blade", "long scalpel", "tweezers"/"fine forceps", "micro scissors", "back micro scissors", "ryder", "short needle holder" (full list in tray.yaml) |
| Pick retry (`pick.py`) | `TRIES` 3; a whole detection within `STILL_THERE_MM` 20 mm of the old spot after lifting = missed |
| Camera "taken" check | red pixels in the lower-middle of the view at the hand-off pose; held #20 = 93,138 px, background 100-1,300 px; taken when below `gone_fraction` 0.3 of the held count for 0.5 s; `min_px` 20,000 (below it: Enter only) |

---

## 5. Repository map

```
arm/                      scrub_arm: driver, kinematics, task layer, HTTP server, teach + calibration tools
  config/arm.yaml         hardware, gains, gripper, pick, hand-off, server (section 4)
  config/poses.yaml       taught joint poses
  scrub_arm/driver.py     motor bus, moves, gripper (set_gripper clamps to [closed, open]), collision guard
  scrub_arm/handoff.py    ScrubAssistant: jobs, _pick_at, _place_at, _present, _let_go
  scrub_arm/server.py     HTTP API on :8765 (section 6)
  scrub_arm/catalog.py    request text -> Resolved (instrument, slot, orientation)
  scrub_arm/calibrate.py  charuco / handeye / touch / aim
  scrub_arm/teach.py      interactive teach tool (jog, gcal, poses)
  tests/                  pytest (one known failure: test_lowering_is_one_straight_move_from_where_centring_left_it)
perception/               scrub_vision: red-parts detector, perception HTTP service on :8770
  config/red_parts.yaml   classes, grasp_offset_mm, scan_q
  config/finger_mask.png  where the fingers are in the image
  scrub_vision/red_parts.py  the detector used by pick.py and the service (--red-parts)
  scrub_vision/service.py    /plan /verify /observe /reset_staging (/count on main)
  tests/                  pytest (all pass)
config/tray.yaml          instruments, aliases, jaw_open_mm, in_set, sutures, slots (shared by arm, voice, CV)
config/camera.yaml        intrinsics, hand-eye, mat, table_z
tools/                    real-arm scripts (tools/README.md lists each); pick.py is the main one
hw_check/                 bus checks (ping_each, clear_faults, disable_each), saved views and calibration captures
voice/                    scrub_voice: standalone push-to-talk + faster-whisper -> last_command.json
sim/, predict/            MuJoCo pile sim / RL policy; OpenAI-vision video watcher that POSTs /handoff
docs/                     REAL_ARM_RUNBOOK.md, this file, superpowers/specs (designs, incl. take-back)
console/, gesture/        ON MAIN ONLY (section 6)
```

Useful tools: `tools/check_ids.py` (scan and report each part's identity; `--frame <stem>` for a saved photo),
`tools/tool_jaw_calib.py` (pick each named tool, log grab width and stall, put back), `tools/pick_test.py <ids>`
(picks with a photo at every step into `hw_check/real/steps/`), `tools/grasp_calib.py X Y YAW` (pick motion onto bare
paper, photograph where the fingertips meet), `tools/capture_views.py OUT SEED N` + `tools/build_finger_mask.py OUT`.

---

## 6. The voice / service layer

### 6.1 Arm HTTP API (`arm/scrub_arm/server.py`)

Start: `cd arm && DYLD_LIBRARY_PATH=/usr/local/lib ../.venv/bin/python -m scrub_arm serve [--sim]` (127.0.0.1:8765).

| Route | Body | Returns |
|---|---|---|
| `GET /status` (`/`) | | `{phase, holding, holding_slot, job, last_job, arm, events}` |
| `POST /resolve` | `{"instrument","suture"?,"size"?,"orientation"?}` or `{"text": "..."}` | 200 `{slot, instrument, orientation, say}`; no motion |
| `POST /handoff`, `/stage` | same | 202 `{job, ...info}` |
| `POST /release` | `{}` | release now, or "will release on arrival" during a handoff job |
| `POST /return`, `/home` | `{}` | 202 job |
| `POST /stop`, `/reset` | `{}` | freeze (motors stay on) / clear |

Errors: unknown or unavailable instrument 400 `{error, say}`; busy 409 "One moment, I'm still moving."; `say` strings
are meant to be spoken. Phases: idle, picking, verifying, returning, moving, presenting, releasing, staged, stopped,
fault. Job status: running, done, stopped, fault, failed.

On **main** the server also needs `Authorization: Bearer $SCRUB_API_TOKEN` (when a token is set; it refuses to bind
off-loopback without one), rejects unknown `Origin` headers, checks the Host header, writes `logs/audit.jsonl`
(HMAC with `SCRUB_AUDIT_KEY`), serves `GET /audit` and `GET /status?since=<unix s>` (20 Hz samples).

### 6.2 Resolving requests (`arm/scrub_arm/catalog.py`)

- Match order: exact phrase, longest contained phrase, then fuzzy (difflib 0.75) only among phrases with the same
  digits, so "11 blade" never becomes "10 blade".
- In `ik` mode (the real arm): the instrument must be in `in_set`, else "we don't have the X in this set"; a bare
  "scalpel" gives the first in-set scalpel (`scalpel_10`); suture and size are parsed but not used.
- Orientation: forehand/backhand words, default forehand for orientable instruments only.

### 6.3 Perception service (`perception/scrub_vision/service.py`)

Start: `cd perception && ../.venv/bin/python -m scrub_vision.service --red-parts` (127.0.0.1:8770). `POST /plan
{"instrument": id}` returns `{action: grasp_target|fail, target: {id, x_mm, y_mm, yaw_deg, thickness_mm, width_mm,
confidence, exposure, ambiguous}, say}`. The arm calls only `/plan` (timeout 5 s, no fail-open). On main: `POST
/count` lists the instruments in view.

### 6.4 Voice

- `voice/` (this branch): push-to-talk, local faster-whisper, resolves with `Catalog(tray)` in **slot mode** and writes
  `last_command.json`. Nothing reads that file; it doesn't call the arm. Its slot-mode catalog disagrees with the
  arm's ik mode ("scalpel" -> scalpel_11 vs scalpel_10; "11 blade" accepted vs refused).
- `console/` (main): browser mic -> Cartesia Ink STT -> Claude intent (strict tools; rule-based fallback with no key)
  -> every instrument checked with the arm's `/resolve` -> `/handoff` etc. through `ArmClient` (Bearer token). Wake
  word "scrub"; "got it / release / thanks" while presenting releases. Run `python -m scrub_console serve`
  (:8780) next to `python -m scrub_arm serve`; `python -m scrub_console demo --agents` runs a simulated arm. Its
  demo rewrites arm.yaml to slot mode. Env names (values in a git-ignored root `.env`): `SCRUB_ARM_URL`,
  `SCRUB_API_TOKEN`, `SCRUB_PERCEPTION_URL`, `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `SCRUB_CLAUDE_MODEL`, and
  others in `console/scrub_console/config.py`. Per its README, no live-key run of Cartesia or Claude has happened.

---

## 7. Gaps between the service path and `tools/pick.py`

These live only in `tools/pick.py` (and its copy in `gesture/picker.py` on main), not in `ScrubAssistant` / the
server. Bringing them across is the main job before voice can drive the real arm reliably:

1. **Detection in-process vs over HTTP.** `pick.py` runs `RedPartsDetector` on its own camera; the server asks the
   perception service. They can't both hold camera 0.
2. **Miss detection and retry.** `pick.py` rescans after lifting and retries up to 3 times. The server tries twice
   only when the target is unreachable; a missed grasp is logged ("holding on anyway") and it presents empty jaws.
3. **Camera release at the hand-off.** `pick.py` opens when the camera sees the tool taken (or Enter). The server
   waits for `POST /release` (tug sensing is off) or times out after 60 s and keeps holding.
4. **Timeout behaviour.** `pick.py` puts the tool back; the server keeps holding.
5. **After release.** `pick.py` goes to `rest`; the server goes to `home`.
6. **Choosing among detections.** `pick.py`'s `find()` prefers an exact identity, then candidates, then the most
   complete view; check that `/plan` makes the same choice.

Designed but not built: taking a tool back from the surgeon and setting it down in a clear spot
(`docs/superpowers/specs/2026-10-25-take-back-design.md`, not yet approved).

---

## 8. Running and testing

| What | Command (from the repo root unless noted) |
|---|---|
| Venv | `.venv` (Python 3.12.4), `pip install -e "arm[hardware,kinematics,calibrate,dev]" -e "perception[dev]"` |
| Bus check (nothing moves) | `DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/ping_each.py` |
| Clear motor faults | `DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/clear_faults.py` |
| Teach tool | `cd arm && DYLD_LIBRARY_PATH=/usr/local/lib ../.venv/bin/python -m scrub_arm teach` |
| Identify parts | `DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/check_ids.py` |
| Arm tests | `cd arm && ../.venv/bin/python -m pytest` (1 known failure, see the repo map) |
| Perception tests | `cd perception && ../.venv/bin/python -m pytest` (all pass) |
| Voice tests | `cd voice && ../.venv/bin/python -m pytest` |

---

## 9. Things that have bitten before

- **Never command the gripper past `closed_deg`.** It snapped once. The clamp in `set_gripper` enforces it; keep it.
- **Jaws never reopen after lifting** except when a tool has been set down or at the hand-off (user rule). The
  grasp's "holding" flag is unreliable, so never open on it.
- **The camera mount moves** when the arm is knocked (it happened during a CAN fault). Symptoms: picks a fixed
  distance off, parts measuring the wrong length, a #10 read as "10 or 20". Fix: redo the hand-eye
  (`scrub_arm.calibrate handeye`, board flat and unmoved), rebuild the finger mask, re-check `grasp_offset_mm`.
- **CAN faults** ("joint N 25 deg off", `disable_all` ack timeouts): usually a loose cable (the joint 1-2 cable was
  reseated once) or the bus held by another process. `ping_each.py` finds the silent motor.
- **Do not change the pick motion** (hover, jaws to width, one straight move down, close, straight up). The user
  had a look-around variant reverted.
- **Parts near the right finger** in the scan view are partly hidden; the cut-off rule handles a single hidden end,
  but keep the tools away from the paper's edges.
- **Hands in view** during a pick move the tools and corrupt the photos.
