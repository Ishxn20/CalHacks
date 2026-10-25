"""Per-tool jaw check after a gripper re-zero: pick each tool, log where the jaws stall on it, put it back.

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/tool_jaw_calib.py "20 blade" "10 blade" tweezers

For each tool: the camera's width at its grab point, the open width it comes down with (tray.yaml jaw_open_mm) and
that width's motor angle, where the jaws stalled on it (motor angle and the table's mm) and whether it was held.
Suggests jaw_open_mm = grab-point width + 2 x 10 mm (the tray.yaml rule). Writes nothing.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick as P  # noqa: E402  (builds the arm app and opens the camera)

g = P.app.cfg.gripper
rows = []
_grasp = P.d.gripper_grasp


def grasp_and_log():
    ok = _grasp()
    rows[-1]["stall_deg"] = math.degrees(P.d.gripper_measured()) - math.copysign(g.squeeze_deg, g.closed - g.open)
    return ok


P.d.gripper_grasp = grasp_and_log

def row(r):
    print(f"  >> {r['name']:24s} grab {r['vision_mm']:5.1f} mm | opens to {r['open_mm']:.0f} mm "
          f"({math.degrees(g.angle_for(r['open_mm'])):.1f} deg) | stalled at {r['stall_deg']:.1f} deg "
          f"= {g.width_at(math.radians(r['stall_deg'])):.1f} mm | held {'yes' if r['held'] else 'NO'} | "
          f"suggested open {r['vision_mm'] + 20:.0f} mm", flush=True)

P.d.start()
P.d.enable()
try:
    for text in sys.argv[1:] or ["20 blade", "10 blade", "tweezers"]:
        r = P.app.catalog.resolve(text)
        x = P.find(P.look(), r.instrument.id)
        rows.append({"name": r.instrument.name, "vision_mm": x.width_mm if x else float("nan"),
                     "open_mm": r.instrument.jaw_open_mm, "stall_deg": float("nan")})
        held = P.pick(r)
        rows[-1]["held"] = held
        row(rows[-1])                        # printed now: a Ctrl-C later doesn't lose it
        if held:
            P.a._place_at(*P.a._picked_from)
    P.d.gripper_open()
    P.d.move_pose("rest", speed=0.5)
finally:
    P.cap.release()
    P.park_and_shutdown(P.app)

print(f"\n  closed {math.degrees(g.closed):.1f} deg, open {math.degrees(g.open):.1f} deg "
      f"({g.width_at(g.open):.0f} mm)")
for r in rows:
    row(r)
