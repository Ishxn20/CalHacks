# scrub_vision: find instruments in a pile

Segments every visible instrument (YOLO11-seg trained on synthetic piles), identifies it by metric length + STL silhouette, decides what's on top, and tells the arm what to pick next. Instruments may be any colour.

```bash
# from the repo root, once
python3.12 -m venv .venv && .venv/bin/pip install -e "arm[dev,kinematics,calibrate]" -e "perception[dev]" mujoco
cd perception
../.venv/bin/python -m scrub_vision.synth --n 3000 --out data/synth                    # random-placement piles
../.venv/bin/python -m scrub_vision.synth --source mujoco --n 1500 --out data/synth_mj  # physics-settled piles
../.venv/bin/python -m scrub_vision.train --data data/synth/data.yaml data/synth_mj/data.yaml --epochs 30
../.venv/bin/python -m scrub_vision.service --weights weights/pile_seg.pt               # :8770
open http://127.0.0.1:8770/debug                                                         # what it sees
```

Real photos (do this as soon as the camera is calibrated): save ~200 scan-pose frames of real piles to `data/real/images`, run `scripts/label_assist.py --weights weights/pile_seg.pt`, fix the labels in Roboflow or Label Studio, export YOLO-seg to `data/real`, then train on all three with `--model weights/pile_seg.pt --epochs 10`.

## Hardware bring-up (in this order)

1. **Pinch test.** Lay a 3 mm print on the planned surface, jog the gripper down in `python -m scrub_arm teach`, `close`. It must report `HOLDING`. If not: soft 5 mm sheet, thicker print, or a fingertip lip.
2. **Contact threshold.** Jog down onto a print and watch `status`; the joint 2/3 lag that appears on touch sets `pick.contact_deviation_deg` in `arm/config/arm.yaml`.
3. **Poses.** Teach `scan` (wrist straight down, ~40 cm above the pile, pile in the upper two thirds of the image), `verify`, `handoff`, `home`, `rest`.
4. **Calibration.** `python -m scrub_arm.calibrate intrinsics <checkerboard photos>` then `python -m scrub_arm.calibrate touch`. Copy the printed table z into `pick.table_z_mm`. Check three marked points land within ±5 mm.
5. **Jaw axis.** At `scan`, solve a yaw-0 grasp and confirm the fingers open across the base +y direction; if they open along +x, flip `jaw_axis_local` in `arm/scrub_arm/kinematics.py`.
6. **Reach.** Keep the pile centre and every `pick.staging_mm` spot within ~350 mm of the base (top-down reach is ~400 mm at hover height).
7. Set `pick.mode: ik`, run `python -m scrub_arm serve` and the vision service, then `python -m scrub_arm send handoff resano`.
