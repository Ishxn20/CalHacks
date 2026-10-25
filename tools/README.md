# Real-arm tools

Run from the repo root with the PCAN library on the path, from a terminal that has camera access:

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/<script>.py ...

Outputs (photos + joint angles) go to `hw_check/real/` (gitignored).

| script | what it does |
|---|---|
| `pick.py [TOOL ...]` | type a tool ("10 blade", "tweezers", "ryder"...): pick it, check with the camera it's off the table, retry up to 3x; then hand off (release on tug) or put back. With names on the command line: pick each, put back |
| `snap_scan.py OUTDIR` | go to the scan pose, save 3 frames + joints, park |
| `draw_targets.py STEM OUT.jpg` | draw each detected part's outline, grab point and jaw line on a saved frame |
| `pick_test.py PART [PART ...]` | scan, pick each named part with its jaw width, photograph every stage (target drawn), put it back |
| `capture_views.py OUTDIR SEED N` | photograph the mat from N random poses around the scan pose (training / finger mask) |
| `refit_red_parts.py` | refit `perception/config/red_parts.yaml` from captured views |
| `consistency.py DIR ...` | how consistently different camera poses place the same unmoved parts |
| `grip_sweep.py` | step the gripper through 11 angles at the scan pose, photograph the fingertips (angle -> mm table) |
| `jaw_check.py` | set each tool's open width in the air and measure the real gap with the camera |
| `grasp_calib.py X Y YAW` | real pick motion onto bare table; photograph where the fingertips meet |
| `touch_table.py X Y YAW` | step the open jaws down onto bare table until contact; report the tool-point height |
| `hover_check.py OUTDIR [DX DY]` / `hover_err.py OUTDIR` | hover 3 cm above each part's grab point (no contact) and measure the miss |
