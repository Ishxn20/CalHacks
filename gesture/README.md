# Left-hand tabletop camera demo

Camera video → Claude recognizes the left-hand pose on the brown table → fixed tool → pickup and handoff.

| Left hand resting palm-down on the brown table | Fixed tool |
| --- | --- |
| All five fingers resting normally, thumb closer to the index | #10 scalpel |
| Four long fingers extended; thumb tucked out of view | #20 scalpel |
| All five fingers extended; thumb spread about 90° from the index | Fine vascular forceps |

`main` knows eight printed instruments. This demo deliberately exposes only these three. These cue-to-tool
associations are preset demo controls, not predictions about a real procedure. Claude actually recognizes
the pose from camera images; there is no keyboard, timer or offline substitute. The right hand is not needed.
The left hand and brown/tan table must be visible. A missing/off-table hand forces `none` in Python even if
the model reports a pose. The wide thumb-index L pose takes precedence over the ordinary five-finger pose;
an uncertain angle returns `none`. This is approximate visual classification, not precise angle measurement.

## Run with simulated arm

From the repository root, using the existing Python 3.12 virtual environment:

```sh
.venv/bin/pip install -r gesture/requirements.txt
.venv/bin/python -m gesture
```

Set `ANTHROPIC_API_KEY` in the root `.env` (or your shell). To use another env file, pass `--env-file PATH`.
The model defaults to `claude-sonnet-5-5`; override with `--model` or `GESTURE_CLAUDE_MODEL`.

Open **http://127.0.0.1:8790**, click **Start camera**, allow camera access, and select the FaceTime camera.
Angle the camera toward the brown table and place your left hand on it. No right-hand signal is needed.
Click **Start recognition**. One three-frame sequence at 90% model confidence selects a tool. **Pick selected tool** starts a simulated
pickup; **Release tool** finishes the simulated handoff. The page always labels a simulated arm.

To make pickups automatic, pause recognition, enable **Pick and present automatically**, then start again.
After a request, lift the left hand briefly before repeating the same pose. A different pose can request
a new tool once the arm is idle, subject to a ten-second cooldown. Only one model call and one robot job run
at a time. Low confidence, errors, unknown cues and stale sessions do not trigger a pickup. The page shows
actual Claude latency. For two independent confirmations instead, add `--confirm 2`; adjust the confidence
threshold with `--threshold` (minimum 0.85).

You can also play a recorded clip in the doctor view. Pausing, seeking, switching cameras, hiding the page,
or stopping recognition invalidates the previous camera session. Images are sent to Claude after Start;
this app does not save camera images or videos to disk. Claude's own account data policies still apply.

The doctor-camera selector supports the Mac camera and iPhone Continuity Camera. It remembers the selected
camera label, respects explicit device selection, and discards late responses from previous camera requests.
Camera errors retain Safari's actual error name/message. A temporarily busy camera is retried once; a missing
iPhone is not silently replaced with the Mac camera.

## Physical arm and second camera

Use the existing real-arm setup/calibration from [the runbook](../docs/REAL_ARM_RUNBOOK.md). The three red
printed tools must be spread out and visible on the calibrated paper surface. Shut down other programs
using the CAN adapter or wrist camera first. Then launch explicitly in physical mode:

```sh
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python -m gesture --real-arm --robot-camera 0
```

Choose the actual wrist camera's OpenCV index, which can differ between Macs. The FaceTime camera is
selected independently in the browser. With the physical arm, leave **Robot view → Server wrist camera**:
preview, detection and hand-off sensing share one server-side camera reader. A browser-only second-camera
preview is also available in simulation; it does not connect that camera to the picker.

All demo code lives in `gesture/`; no existing arm, perception, voice, or tool files are modified. The private
`gesture/picker.py` adapter follows `tools/pick.py` at `ccd58d88`, using the shared arm driver, detector and
calibration. It keeps the three camera-checked retries, presentation, camera-based taken detection and timeout
return. It reads handoff calibration from `tools/handoff_calib.json` when present. **Release tool** replaces the CLI's Enter.
**Stop arm & recognition** cancels pending recognition and stops motion. A stopped arm holding a tool must
be recovered with the existing teach tool before Reset; the dashboard does not discard that held-tool state.

The app binds only to loopback and checks Host/Origin on every request. Images go to Claude from Python;
the API key is never sent to the browser. Recognition starts paused and automatic motion starts unchecked.

## Checks

```sh
PYTHONPATH=.:arm:perception .venv/bin/python -m pytest gesture/tests -q
node --test gesture/tests/camera.test.cjs
```

Automated tests use fake model responses, a simulated robot and stubbed physical hardware. They check
image/schema validation, confirmation/cooldown, duplicate suppression, cancellation during a model call,
busy/stop handling, origin checks, and the actual picker integration. Live camera recognition and physical
pickups are separate checks; passing these tests alone does not establish either.
