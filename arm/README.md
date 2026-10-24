# scrub_arm: arm control for the robotic scrub assistant

This package controls the reBot Arm B601-RS. It takes a parsed voice command, picks the right 3D-printed instrument off the tray, has the tray-cam confirm it, and presents it to the surgeon with the wrist set for **forehand** or **backhand**.

```
voice pipeline ──POST /handoff {instrument, suture, size, orientation}──▶ scrub_arm server
                                                                          │
     catalog: "needle holder, 6-0 Prolene" ──▶ slot A1                    │
     driver:  hover → open → descend → grasp (did the jaws stop on it?) → lift
     verify:  POST tray-cam service {expected, slot} → ok?                │  (wrong tool → put it back, report)
     present: swing to handoff pose, joint6 rotated 0° / 180° for forehand / backhand
     release: "release" (voice/API) or a tug on the instrument → open → home
```

Motion uses **taught joint-space poses** replayed with smooth min-jerk trajectories. The tray has fixed slots, so there's no IK to tune and no camera calibration to get wrong. Every move is something you've already watched the arm do.

## Setup

```bash
conda create -y -n rebot python=3.12 && conda activate rebot
pip install -e "arm[hardware,dev]"      # motorbridge has no Intel-Mac wheel; use Ubuntu or Apple Silicon
```

Before running this, do the hardware bring-up in the [Seeed quick start](https://wiki.seeedstudio.com/rebot_b601_rs_getting_started/): motor IDs 1–7, zero calibration, and **Apply Default Template → Write Parameters** in MotorBridge Studio. Then bring CAN up (`sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up` on Linux).

Everything runs without hardware too. Add `--sim` to any command.

```bash
cd arm
python -m pytest                       # 39 tests, full handoff flows against the simulator
python -m scrub_arm serve --sim        # fake arm + real API on :8765
python -m scrub_arm send handoff "needle holder, 6-0 prolene, backhand"
python -m scrub_arm send release
```

## Bring-up on the real arm

1. **`python -m scrub_arm check`**: pings all 7 motors and prints joint angles. Compare the angles with MotorBridge Studio. They should match and read about 0 at the zero pose.
2. **`python -m scrub_arm teach`**, then type `help`. The motors start **disabled**.
3. **Gripper.** `hold`, then use `g <deg>` to find the jaw endpoints. With the jaws fully open, run `gcal open`. With them closed on nothing, run `gcal closed`. Both are written to `config/arm.yaml`. Then `close` with an instrument in the jaws should print `HOLDING`, and `close` with empty jaws should print `empty`. If not, tune `grip_kp` and `holding_min_gap_deg`.
4. **Poses.** Teach each one as follows. **Support the arm first: limp means no torque.**
   - `limp`, then drag the gripper roughly into place.
   - `hold`.
   - `jog <joint> <deg>` to fine-tune.
   - `save <name>`.

   Always save while holding, not while limp. That stores the *setpoint*, so the replay sags exactly as it did while you taught it.
   - `home`: clear of the tray, neutral.
   - `<slot>.above` / `<slot>.grasp` for each slot in `config/tray.yaml` at the repo root (A1…B4). Make `above` high enough that swinging between slots clears everything.
   - `handoff`: where the surgeon's hand will be. **Teach it with joint6 near −90°** so the +180° backhand rotation stays inside the joint limits.
   - `staged` (optional): hold-ready spot for predicted instruments. `verify` (optional): instrument in tray-cam view. `rest`: folded pose the arm parks in before shutdown.
5. `check` lists anything still missing. Then `test needle holder, 6-0 prolene, backhand`, followed by `release`.
6. Raise `motion.speed_scale` gradually once everything is reliable.

The real arm **refuses** to move to placeholder poses (`taught: false`) or to grasp with an uncalibrated gripper. Only the simulator uses those placeholders.

## API (for the voice and CV teammates)

`python -m scrub_arm serve` listens on `127.0.0.1:8765`. Responses are JSON. Errors carry a `say` field you can send straight to TTS.

| Call | Body | Notes |
|---|---|---|
| `POST /handoff` | `{"instrument","suture"?,"size"?,"orientation"?}` | 202 + job. 400 if unknown or not on the tray ("No needle holder loaded with vicryl; have 4-0 prolene, 6-0 prolene"). 409 if busy. |
| | `{"text": "needle holder, 6-0 prolene, backhand"}` | Fallback if the LLM parser is down |
| `POST /resolve` | same | Dry run: which slot, plus the `say` confirmation string. No motion. |
| `POST /release` | – | Surgeon has it. Latched: if said mid-swing, it releases on arrival. |
| `POST /stage` | same as handoff | Prediction stretch: pick the predicted tool and hold it at `staged`. A later `/handoff` for the same tool skips the pick. For a different tool, it puts the staged one back first. |
| `POST /stop` | – | Freeze in place (motors stay on, nothing drops). `POST /reset` to resume. |
| `POST /return`, `POST /home` | – | Put the held tool back / go home |
| `GET /status` | – | `phase` (idle/picking/verifying/presenting/…), `holding`, current/last job, joint angles, recent events |

Field normalization is forgiving. `castro`/`Castroviejo`, `six-oh`/`6/0`/`60`, `polypropylene`→Prolene, and `backwards`→backhand all work. A generic `needle holder` picks whichever needle holder slot is loaded with the requested suture and size. Numbers are never fuzzy-matched: "11 blade" can't become "10 blade". The vocabulary and tray contents live in the repo-root `config/tray.yaml`, shared with `voice/`. Edit it to match what's physically loaded.

**Tray-cam contract (CV teammate).** Set `verify.url` in `arm.yaml`. While holding the tool at `verify`, the arm sends `POST {"expected": "<instrument id>", "slot": "A1"}` and expects `{"ok": bool, "detected": "<id>", "confidence": float}`. With `fail_open: true`, an unreachable service doesn't block the demo.

## Safety

- **Collision guard.** If any joint lags its setpoint by more than `tracking_error_limit_deg`, the arm freezes where it is and the job faults. Clear it with `POST /reset`.
- **`/stop` freezes, it doesn't cut power.** Cutting torque on this arm drops it. Ctrl-C or `kill` on the server parks the arm at `rest` before disabling.
- **Limits.** Joint limits come from the URDF, and every target is checked before moving. Grip force is limited by `grip_kp`, so the printed instruments don't get crushed.
- **Keep the official distance.** Stay at least 1 m away while it runs, and keep a hand near the 48 V supply switch.

## Things to verify on hardware (the simulator can't)

- **Active reporting.** `active_report: true` asks the motors to stream status so measured positions stay fresh. If the collision guard false-trips or the joint readings in `status` freeze while moving, try setting `tracking_error_limit_deg: null` first.
- **Tug-to-release** (`handoff.tug`) is off by default. Turn it on and tune `deviation_deg` by pulling on a presented instrument while watching `status`.
- **Gravity sag in MIT mode** is a few degrees on joints 2–3. Teaching while holding cancels it out. If grasps are still off, try `arm_mode: pos_vel` (the motors' own position loop with integral action) and re-teach.

## Layout

```
config/arm.yaml     motors (IDs/gains from Seeed's rebotarm_rs.yaml), limits, speeds, gripper, handoff, verify
../config/tray.yaml instrument vocabulary, suture names, slot → instrument/suture/size → poses
config/poses.yaml   taught poses (written by `teach`)
scrub_arm/bus.py        motorbridge RobStride backend + simulator
scrub_arm/driver.py     control loop thread, min-jerk moves, gripper, collision guard
scrub_arm/catalog.py    intent → slot resolution
scrub_arm/handoff.py    pick / verify / present / release job runner
scrub_arm/server.py     HTTP API
scrub_arm/teach.py      interactive teaching CLI
```
