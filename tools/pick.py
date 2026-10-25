"""Type a tool, the arm picks it up; if the camera still sees it on the table afterwards, it tries again.

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/pick.py              (interactive)
    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/pick.py "10 blade"   (pick it, then put it back)

Names: anything tray.yaml knows (ids, names, aliases: "10 blade", "tweezers", "ryder", "long scalpel", ...).
Each try: scan pose -> find the tool (red_parts) -> pick (its jaw width, the real pick motion) -> back to the scan
pose and look: if the tool is still lying where it was, that was a miss, so open and try again from a fresh scan
(it may have been nudged). Once it's held: [h]and off (present; when the wrist camera sees the tool has been taken,
or Enter is pressed, open and go to the rest pose; not taken in 60 s: put it back) or [p]ut it back.
"""
import sys
from pathlib import Path

R = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
sys.path.insert(0, R + "/perception")
from scrub_arm.app import build                      # noqa: E402
from scrub_arm.camera_pick import CameraPickup       # noqa: E402
from scrub_arm.catalog import ResolveError           # noqa: E402
from scrub_arm.teach import park_and_shutdown         # noqa: E402

# Module-level so the calibration scripts (handoff_calib, handoff_view, tug_calib) can `import pick as P`.
app = build()
d, a = app.driver, app.assistant
camera = CameraPickup(a)
pick, held_red, wait_until_taken = camera.pick, camera.held_red, camera.wait_until_taken


def after_pick(interactive: bool) -> None:
    choice = input("  [h]and off or [p]ut back? ").strip().lower()[:1] if interactive else "p"
    if choice == "h":
        a._present(a.holding)
        d.sleep(0.5)
        print("  presenting: take the tool (or press Enter to release)", flush=True)
        how = wait_until_taken(interactive)
        if how == "timeout":
            print("  not taken; putting it back")
            a._place_at(*a._picked_from)
            return
        a._let_go()                       # open to the tool's width
        d.move_pose("rest", speed=0.5)
        print(f"  handed over ({'the camera saw it taken' if how == 'taken' else 'Enter'}); at rest")
    else:
        a._place_at(*a._picked_from)
        print("  put back")


def main() -> None:
    names = sys.argv[1:]
    d.start()
    d.enable()
    try:
        while True:
            text = names.pop(0) if names else (None if sys.argv[1:] else input("tool> ").strip())
            if not text or text in ("q", "quit", "exit"):
                break
            try:
                r = app.catalog.resolve(text)
            except ResolveError as e:
                print(f"  {e}")
                continue
            d.clear_abort()
            if pick(r):
                after_pick(interactive=not sys.argv[1:])
    finally:
        camera.close()
        park_and_shutdown(app)


if __name__ == "__main__":
    main()
