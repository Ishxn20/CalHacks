# Scrub: a robotic surgical scrub assistant

Scrub is a robot arm that acts as a scrub nurse. You ask for an instrument by voice, by a hand signal or by typing.
The arm finds the instrument on the table with its wrist camera, picks it up and hands it to you. It lets go once
the camera sees that you have taken it. A web console tracks where every instrument is, runs the end-of-case count
and keeps a tamper-evident record of what happened. The console runs on Convex.

Built at Cal Hacks 13.0. The build log and the Convex evidence are in [hackathon.md](hackathon.md).

## How we use Convex

The operating-room console runs on Convex: a live instrument board, an event timeline and arm charts.
Recorded hand-off episodes go to Convex file storage, and Claude labels them in a scheduled action.
Crons retry failed labels and prune old readings, and an HTTP action exports the accepted dataset.
See [hackathon.md](hackathon.md) for the features, the files that show them, and the build log.

## System overview

```
 voice (Whisper / Cartesia) ─┐
 hand signal (Claude vision) ┼─► console ── Claude intent ──► arm server :8765 ──► reBot B601-RS (CAN)
 typed request ──────────────┘   :8780  │                        │  ▲
                                        │                        ▼  │ wrist camera
                                        ▼                perception :8770 (find + identify parts)
                         Convex (events, instruments, 20 Hz readings, episodes, audit heads)
                                        │
                         dashboard ◄── live queries (ConvexClient.onUpdate)
```

Every request goes through the arm's own catalog check (`POST /resolve`) before anything moves. Claude never moves
the arm directly: if it names an instrument that isn't on the tray, the console asks a question instead.

## Hardware

| Part | Spec |
|---|---|
| Arm | Seeed reBot Arm B601-RS: 6 joints plus a gripper, RobStride motors (`rs-06` on joints 1-3, `rs-00` on joints 4-6), motor IDs 1-7 |
| Bus | CAN at 1 Mbit/s through a PCAN-USB adapter (MacCAN PCBUSB library on macOS) |
| Power | 48 V supply. Keep a hand near the switch and stay at least 1 m away while the arm moves. |
| Control loop | 200 Hz setpoint streaming with min-jerk trajectories and a 20 Hz readings buffer for the dashboard |
| Joint limits | J1 ±160°, J2/J3 0-179°, J4/J5 ±89°, J6 ±180° (from the URDF, `arm/config/arm.yaml`) |
| Gripper | Calibrated 0-95.5 mm jaw gap (~0.33 mm per degree). It closes until the jaws stall on the tool, then squeezes 4° more and never goes past `closed_deg`. |
| Wrist camera | USB camera on the gripper, 1920x1080, OpenCV index 0. Hand-eye calibrated: intrinsics rms 0.21 px, hand-eye 1.7 px over 7 views. |
| Work area | Parts lie spread out on white paper, about 380-560 mm in front of the base |
| Instruments | Red 3D prints made from the STL models in `perception/assets` |

## Instruments the camera recognises

| Say or type | Part | Grabbed at |
|---|---|---|
| `10 blade`, `#10` | #10 scalpel (~153 mm) | centre |
| `20 blade`, `#20` | #20 scalpel (~171 mm) | centre |
| `long scalpel` | long #11 scalpel (~236 mm) | centre |
| `tweezers`, `fine forceps` | fine vascular forceps (~187 mm) | centre |
| `forward micro scissors` | forward micro scissors (~156 mm) | thinnest point |
| `back micro scissors` | back-facing micro scissors (~156 mm) | thinnest point |
| `ryder` | Ryder needle holder (~185 mm) | neck |
| `short needle holder` | short needle holder (~155 mm) | neck |

The full vocabulary is in `config/tray.yaml`, which voice, arm and vision share. That includes the slot-mode tray
instruments, suture names and aliases such as `castro` and `six-oh`. Anything else on the table is classed as
"other" and is moved aside.

## How one pick works

1. **Scan.** The arm moves to the taught `scan` pose and photographs the table.
2. **Find.** `perception/scrub_vision/red_parts.py` thresholds red in Lab space, then maps each pixel to millimetres
   on the table using forward kinematics, the hand-eye calibration and the camera intrinsics. It names each part by
   six shape features (length, mean width, median width, narrow-end width, taper, solidity). Needle holders are
   named by their ring holes instead. On held-out real frames it named 240 of 241 parts correctly.
3. **Pick.** The arm hovers 10 cm above the grab point with a 45° gripper tilt. It opens the jaws to the tool's
   `jaw_open_mm`, moves straight down, grasps and lifts. If a joint would be near its limit, it comes in rotated 180°.
4. **Check.** The arm rescans the table. If the tool is still there, it retries, up to 3 times.
5. **Hand off.** The arm presents the tool. It opens the jaws half a second after the camera sees the tool taken,
   or puts the tool back after 60 s.

## Software components

| Directory | Package | What it does | Main tech |
|---|---|---|---|
| `arm/` | `scrub_arm` | Motor driver, control loop, collision guard, IK picks, hand-off jobs, HTTP API, teach CLI, calibration | motorbridge, Pinocchio (`pin`), OpenCV |
| `perception/` | `scrub_vision` | Instrument detection and identification, declutter planner, `/plan` `/verify` `/count` `/debug` service | Ultralytics YOLO11-seg, OpenCV, trimesh |
| `console/` | `scrub_console` + `convex/` | Operating-room console: intent, voice, dashboard, instrument board, count check, audit anchors, fault triage, episode recording | Convex, Anthropic, Cartesia, Fetch.ai uAgents, Factory Droid SDK |
| `voice/` | `scrub_voice` | Push-to-talk speech to an instrument id, written to `last_command.json` | faster-whisper (`base.en`, `small.en`) |
| `gesture/` | `gesture` | Left-hand pose on the table to a fixed tool, then pick and hand-off | Claude vision (`claude-sonnet-5-5`) |
| `predict/` | `watch.py` | Watches a surgery video and stages Potts scissors before they're needed | OpenAI vision (`gpt-6.1-sol`, `gpt-6-luna`) |
| `sim/` | | MuJoCo pile generator, declutter stress test, saved RL grasp-policy snapshots | MuJoCo |
| `tools/` | | Real-arm scripts: `pick.py`, calibration, grip sweep, hand-off calibration | |
| `hw_check/` | | Motor ping, clear faults, per-joint diagnostics | |

Python 3.10+ is required (3.12 in practice). The console's Convex functions are TypeScript (`convex` 1.48,
`@anthropic-ai/sdk`, `zod`), tested with `convex-test` and `vitest`.

## Convex backend (`console/convex/`)

| Table | Holds | Indexes |
|---|---|---|
| `events` | Every utterance, intent, command, job, fault and count | `by_kind` |
| `instruments` | One row per physical instrument: tray, held, in field or used | `by_uid` |
| `metrics` | Raw 20 Hz arm readings, one document per ~5 s batch | `by_t0` |
| `metricPoints` | The same readings rolled up to one point per second. Each point keeps the worst lag and torque in its second. | `by_t0` |
| `episodes` | Recorded arm jobs: face-blurred video and frame tables in file storage, plus Claude labels and a review gate | `by_labelStatus`, `by_review` |
| `anchors`, `alerts` | Audit-log chain heads from the arm and the console, and rollback or fork alerts | `by_source` |
| `proposals` | Fault-triage proposals and their approval status | `by_proposal_id` |

- **Writes** are internal mutations. Only the console can call them, over Convex's HTTP API (`console/scrub_console/sinks.py`).
- **Public queries** feed the dashboard: `instruments:list`, `events:recent`, `events:byKind`, `metrics:latest`,
  `metrics:points`, `metrics:raw`, `episodes:list`, `episodes:stats`, `audit:latest`, `audit:alerts`, `proposals:recent`.
- **Scheduled action** `labeling:labelEpisode` has `claude-opus-5-5` write structured labels for each episode,
  including privacy flags.
- **Crons:** failed labels are retried every 10 minutes, and arm readings older than a day are pruned hourly.
- **HTTP action:** `GET /export/manifest` returns accepted episodes with download URLs. It needs a bearer export key
  and fails closed when the key isn't set.
- **Live screens:** the Live tab's instrument board, event timeline and latest readings, and the Robot data tab's
  log view, all update without a refresh.

## Ports and APIs

All services bind to loopback (127.0.0.1).

| Port | Service | Key endpoints |
|---|---|---|
| 8765 | Arm server (`python -m scrub_arm serve`) | `POST /handoff`, `/resolve`, `/release`, `/stage`, `/stop`, `/reset`, `/return`, `/home`; `GET /status` |
| 8770 | Perception (`python -m scrub_vision.service`) | `POST /plan`, `/verify`, `/count`; `GET /debug` |
| 8780 | Console and dashboard (`python -m scrub_console serve` or `demo`) | Web dashboard, server-sent events |
| 8790 | Gesture demo (`python -m gesture`) | Web page with camera preview |

The arm server's errors carry a `say` field that can go straight to text-to-speech.

## Safety and security

- **Collision guard.** If a joint lags its setpoint by more than `tracking_error_limit_deg`, the arm freezes and
  the job faults.
- **`/stop` freezes the arm, it doesn't cut power.** Shutting the server down parks the arm at `rest` before the
  motors are disabled.
- **Gripper limits.** The jaws never open in mid-air while holding a tool and never close past the calibrated limit.
- **Locked-down arm API.** It needs a bearer token (`SCRUB_API_TOKEN`), allows no wildcard CORS and checks Origin and
  Host. It keeps a hash-chained, HMAC-signed audit log (`logs/audit.jsonl`), and the chain heads are published to
  Convex so a truncated log is noticed.
- **Fault triage is gated.** Droid only edits a sandbox copy of `arm.yaml`. Its diff is checked against an allowlist
  of tuning keys and bounds, and a person must approve it.
- **Simulated results are always labelled**, including the simulated arm, the simulated count and rehearsal triage.

## Quick start

No hardware needed:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e "arm[kinematics]" -e "console[all]"
python -m scrub_console demo --agents        # simulated arm + console + dashboard on http://127.0.0.1:8780
```

With the real arm (see [the runbook](docs/REAL_ARM_RUNBOOK.md) first):

```bash
.venv/bin/pip install -e "arm[hardware,kinematics,calibrate,dev]" -e "perception[dev]"
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/pick.py     # type a tool, the arm picks it
```

Copy `.env.example` to `.env` at the repo root and fill in your own keys. `.env` is git-ignored.

## Tests

```bash
cd arm && python -m pytest
cd perception && python -m pytest
cd voice && .venv/bin/python -m pytest
cd console && python -m pytest
cd console && npm install && npm test && npm run typecheck      # Convex functions
PYTHONPATH=.:arm:perception .venv/bin/python -m pytest gesture/tests -q
```

## Known limits

- The paid services (Cartesia, Bitdeer, Droid) have only been tested against fakes, not live keys.
- The camera count is tested on rendered scenes, not real photos.
- Picks depend on the hand-eye calibration. If the camera is knocked, re-measure the grab-point correction
  ([runbook](docs/REAL_ARM_RUNBOOK.md)).
- Tug-to-release via joint torque is too weak to use, so hand-off detection relies on the camera.

## More documentation

- [arm/README.md](arm/README.md): arm API, teaching poses, safety
- [perception/README.md](perception/README.md): training and running the vision service
- [console/README.md](console/README.md): console, sponsor integrations, arm readings
- [console/docs/PRODUCTION.md](console/docs/PRODUCTION.md): production framing
- [docs/REAL_ARM_RUNBOOK.md](docs/REAL_ARM_RUNBOOK.md): real-arm setup, calibration and recovery
- [gesture/README.md](gesture/README.md), [voice/README.md](voice/README.md), [tools/README.md](tools/README.md)
- [hackathon.md](hackathon.md): Cal Hacks 13.0 build log
