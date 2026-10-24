# CLAUDE.md: CalHacks robotic scrub assistant

Context for anyone, human or agent, picking up this repo. It covers what the project is, how the hardware behaves, how this Mac is set up, what each branch holds, and what has and hasn't been verified on the real arm.

## Project

A robot arm acts as a cardiac-surgery scrub tech. It hears an instrument request ("needle holder, 6-0 Prolene, backhand"), picks the 3D-printed instrument, and hands it over in the right orientation. A camera confirms or finds the right tool. The original plan is `Surgical Assistant Idea.pdf`. The later "Scrub bot build plan" adds a clean zone and a used zone, a handle-first handoff, a return path, and an instrument count. It says to use classic CV, IK and taught poses, not RL.

Team:
- **Aditya Mahesh:** arm control (`arm/`).
- **Ishan Vaish:** gripper limits, GPU training of the YOLO detector.
- **Varun Pabreja:** voice agent (`voice/`), YOLO detector (`vision/`), classic object detection (`object_detection` branch).
- **Aditya C (camelCasedAditya):** RL and simulation (`sim/`, `perception/`).

## Hardware: Seeed reBot Arm B601-RS

- **Motors:** 7 RobStride motors on one CAN bus at 1 Mbps, through a PEAK PCAN-USB adapter (USB vendor 0x0C72, shows as "XCAN-USB"). The DIP switch must be on **120R**; BOOT is for firmware flashing only.
- **IDs and models:** joints 1–3 are motors 0x01–0x03, model rs-06. Joints 4–6 are 0x04–0x06, model rs-00. The gripper is 0x07, rs-00. Feedback/host ID is 0xFD for all.
- **Power:** 48 V supply. **Cutting torque (limp or disable) makes the arm drop. Support it first.**
- **SDK:** `motorbridge` 0.5.6. `request_feedback()` is a no-op on RobStride. Positions come from MIT reply frames or active reporting (comm_type 24, which we enable). A reliable read is param `0x7019` (mechPos).
- **Modes and gains:** MIT, with gains copied from Seeed's `reBotArm_control_py/config/rebotarm_rs.yaml`. RobStride only accepts a run-mode change while disabled, so the order is: set mode, then enable.
- **Joint limits:** from Seeed's URDF (`arm/urdf/ReBot_Arm_RS.urdf`). Joint 1 ±160°, joints 2 and 3 0–179°, joints 4 and 5 ±89°, joint 6 ±180°.
- **Kinematics:**
  - **Measured:** at the zero pose (folded), FK puts the tool point (`gripper_end`) at x≈302 mm, z≈218 mm. A tape measure gave 292 and 229 mm, so FK agrees to about 1 cm.
  - **Pointing down:** joints 2, 3 and 4 pitch about parallel axes, so the gripper points straight down when **q2 − q3 − q4 = 90°**.
  - **Only reachable low:** pointing straight down works only with the tool point within about 10–15 cm of the table, and about 10–40 cm out from the base (joint 4 runs out of range higher up).
  - **Table:** the table is at about z = 0 in the base frame (measured table_z = −0.6 mm on the wood).
- **Gripper:** a rotary rs-00 with about **260° of travel** (Ishan calibrated open at 259.42°, closed at −0.20°). Jaw opening is about 0.3 mm per degree. The gripper code **must** ramp the setpoint and cap torque, or the jaws slam. Seeed's own grasp code closes in torque mode (kp = 0, kd 0.5, about 1 Nm). A stall after real travel means holding; reaching the hard stop means a miss.
- **Wrist camera:** "icspring camera", USB, 1920×1080, mounted on top of the gripper and **looking along the gripper**. The gripper's own black jaws fill the bottom centre of its image. It gives black objects a blue tint. On macOS, camera index numbers shuffle when devices are replugged, so run `python -m scrub_arm cams` to find it. At one point it was unplugged, which gives black frames.

## This Mac (Apple M5)

- **Python:** the system and Homebrew Pythons are **Intel builds under Rosetta**, and `motorbridge` has no wheel for them. Use the native env instead: **Miniforge at `~/miniforge3`, env `rebot` (Python 3.12, arm64)**. It was installed in batch mode, so there's no shell init:
  ```bash
  source ~/miniforge3/bin/activate rebot
  ```
- **PCAN driver:** PCBUSB is installed user-local in `~/.local/lib`, with a `PCBUSB` symlink. The env's `etc/conda/activate.d/pcbusb.sh` exports `DYLD_FALLBACK_LIBRARY_PATH`.
- **Installed in `rebot`:**
  - **motor and core:** motorbridge, numpy, pyyaml, pytest;
  - **vision:** opencv-python, ultralytics and torch (MPS works);
  - **simulation and RL:** mujoco, stable-baselines3, gymnasium, pin (pinocchio), imageio, scipy, trimesh;
  - **editable repo packages:** `arm` and `perception` (whichever checkout is current).
- **One program at a time:** only one process can open the PCAN adapter on macOS. Quit `teach` before running `track`, and so on.
- **Camera permission:** VS Code and Terminal need it under Privacy & Security → Camera.
- **"Motors not responding" / ping timeouts:** usually the 48 V power is off or the CAN cable is loose. Re-plugging the USB adapter or power-cycling the arm has fixed it before.
- **Git:** the `origin` remote is HTTPS (`https://github.com/Ishxn20/CalHacks.git`) via `gh auth setup-git`; there's no SSH key. Fetches sometimes die with "early EOF", so retry with `GIT_TERMINAL_PROMPT=0 git fetch`.

## Branches (as of 2026-10-24)

| Branch | What | Status |
|---|---|---|
| `main` | `arm/` control (PR #1), `voice/` agent (PR #2), gripper limits | merged |
| `arm-grab-vision` | top-down grab, numpy FK/IK, gripper torque fix, `track` camera tool | **PR #3 open** |
| `vision-training` / `results/gpu` | Varun's YOLO detector, trained on STL renders; Ishan's trained `best.pt` (mAP50 0.995 on synthetic data) | unmerged |
| `claude/rl-grasp-policy-f51e28` | Aditya C's MuJoCo sim + SAC reach policies (`sim/`) and the `perception/` pile segmenter | unmerged; currently checked out |
| `object_detection` | Varun's classic (non-ML) detection | not yet reviewed |
| `test-vision` (local only) | `arm-grab-vision` merged with `results/gpu`, for testing the YOLO finder | not pushed |

Note: `arm-grab-vision` and the RL branch both changed `arm/` independently, so expect conflicts when merging.

## `arm/` package (`scrub_arm`): CLI `python -m scrub_arm {serve,teach,track,cams,check,send}`

**Config files:**
- **`config/arm.yaml`:** motors, limits, motion and gripper caps, and the `grab:` section, which includes `table_z_m` and a hard floor `floor_z_m: -0.01`.
- **`config/poses.yaml`:** taught joint poses. `taught: false` marks a placeholder, and **the real arm refuses placeholder poses and an uncalibrated gripper**.
- **`config/tray.yaml`:** at the repo root on newer branches; the instrument vocabulary.
- **`config/vision.yaml`:** gitignored, per machine.

**Modules:**
- **`driver.py`:** control-loop thread at 200 Hz, min-jerk and Cartesian `follow_path`, collision guard (tracking error over 25° freezes the arm), torque-capped gripper ramp, torque-mode `gripper_grasp`.
- **`grasp.py`:** `grab_below()`: a straight vertical descent that stops at table + 1 mm, on contact (joint deflection over 2°), or at the floor. It backs off 1 mm, grasps, and lifts, adapting the lift height (80 → 40 → 20 mm). `point_down()` uses seeds that satisfy q2 − q3 − q4 = 90°.
- **`kinematics.py`:** URDF-based numpy FK, Jacobian, DLS IK and `line()`.
- **`vision.py` / `visual_grasp.py` / `track.py`:**
  - **Detection:** click-to-learn HSV colour detection. It has a saturation floor; grey objects are matched by brightness, and blobs over 15% of the image are ignored.
  - **Calibration (`c`):** image-based visual servoing calibrated in one step: the target pixel, plus two 2 cm test moves that give the Jacobian.
  - **`g`:** look along the gripper while sweeping the base, aim, cast the line of sight onto the table, approach, rotate in place to point down, centre, align the jaws across long objects, grab.
  - **Other keys:** `p` parks, `s` stops, and `t` needs a double press.
- **`handoff.py` / `server.py`:** slot-based pick → verify → present (forehand/backhand via a joint-6 offset) → release. HTTP API on :8765 for the voice pipeline.

**Hardware results:**
- **Worked:** `teach` → `table` → `down` → `grab` picked up a rubber band.
- **Didn't work:** `track` failed. A bad colour click matched background greys. A stray `t` press set the table height to +50 mm (it has been reset). A dark clamp can't be told apart from the black jaws, cables and bags by colour. Fixes: give each instrument a bright colour or tape, mask out the jaw region, or use ArUco markers.
- **Unknowns:** `jaw_axis_deg` is unverified.

## Other components

- **`vision/`** (YOLO finder):
  ```bash
  python -m scrub_vision.find --weights results/gpu/best.pt --image X.jpg --instrument clamp_aortic_cross
  ```
  Live mode watches `last_command.json`. The `--weights` flag is required because the default path doesn't exist. It's trained only on turquoise top-down renders, so it's unverified on real photos.
- **`sim/`** (RL, MuJoCo), run from the repo root:
  - `python -m sim.watch_arm --episodes 2 --camera sim/policy/reach_R5_best.zip` is the main demo: blind SAC reach, then camera-guided residual SAC (IBVS plus learned residual), then descent and grasp.
  - `python -m sim.eval_reach --visual --residual sim/policy/reach_R5_best.zip --n 50` reproduces the success-rate comparison (reported 99–100% in sim).
  - The 46 tests pass once `scipy` is installed; it's missing from `sim/requirements.txt`.
- **Real-arm RL path:** `arm.yaml` `pick.mode: ik` + `pick.reach: rl` plus the `perception` service on :8770. It's blocked: there are no `perception/weights/pile_seg.pt`, no `config/camera.yaml`, and no taught `scan`/`home`/`handoff`/`rest` poses on that branch. The bring-up order is in `perception/README.md`.

## Working notes

- **Moving the arm:** always use `teach` or `track`. `hold` before any motion. `go rest` (in `teach`) or `p` (in `track`) parks the arm folded.
- **Checks before testing on hardware:**
  1. `python -m scrub_arm check` shows all 7 motors OK.
  2. Clamp the base and keep about 1 m clear.
  3. Keep a hand on the 48 V switch.
- **Tests:** `cd arm && python -m pytest`. They run against `SimBus`, which has knobs for `object_in_jaws`, `external_offset` and `arm_blocked`, plus `SimCamera`.
- **Committing:** only when asked. Branch off `main`, then open a PR. Attribution is per the Claude Code commit guidelines.
