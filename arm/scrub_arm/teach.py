"""Interactive pose teaching.

Workflow per tray slot:
    limp            -> support the arm and drag the gripper roughly over the slot
    hold            -> motors hold it there
    jog 2 -1.5      -> fine-tune with small joint nudges
    save A1.above   -> stores the *setpoint*, so replay reproduces the same gravity sag
    ...lower to grasp height, center the jaws on the grip tab...
    save A1.grasp

Saving the commanded setpoint (not the dragged position) is deliberate: the
joints sag a little under load in MIT mode, and taught-while-holding poses
replay with identical sag, so the jaws land where you saw them land.
"""
from __future__ import annotations

import math
import shlex

import numpy as np

from .app import App, check_poses
from .catalog import ResolveError, parse_freeform
from .config import save_gripper_calibration, save_table_z
from .grasp import grab_below, point_down, tilt_from_down
from .handoff import Busy

HELP = """\
  s | status              joint angles (deg), gripper, phase
  limp                    cut torque to move the arm by hand (SUPPORT IT FIRST - it will drop)
  hold                    re-energize, holding the current position
  jog <joint 1-6> <deg>   nudge one joint (setpoint), e.g. jog 2 -1.5
  set <joint 1-6> <deg>   move one joint to an absolute angle
  save <name>             save current setpoint as a pose (A1.above, A1.grasp, home, handoff, staged, verify, rest)
  go <name> [speed]       move to a saved pose (speed multiplier, default 1)
  poses                   list poses
  del <name>              delete a pose
  open | close            gripper open / soft close (close reports whether it's holding something)
  pose                    tool-point position (mm) and how far the gripper is from pointing straight down
  down                    tilt the wrist so the gripper points straight down (tool point stays put)
  table                   touch the fingertips to the table, then `table` to record its height
  grab [depth_mm]         grab whatever is directly below: descend, close, lift (stops on contact)
  g <deg>                 gripper to an absolute motor angle
  gcal open|closed        record the current gripper angle as the open / fully-closed endpoint
  check                   list placeholder/missing poses and other setup problems
  test <request>          full handoff, e.g.: test needle holder, 6-0 prolene, backhand
  stage <request>         pick and hold at the staged pose
  release                 surgeon has it: open and go home
  return                  put the held instrument back in its slot
  stop / reset            freeze / clear a stop or fault
  q                       quit (moves to 'rest' if taught, then goes limp)"""


def _fmt(q) -> str:
    return " ".join("  nan " if np.isnan(x) else f"{math.degrees(x):+7.2f}" for x in q)


def run_teach(app: App) -> None:
    d, poses, asst = app.driver, app.poses, app.assistant
    d.start()
    print("Connected. Motors are DISABLED. Type `hold` to energize where the arm is now, `help` for commands.")
    while True:
        try:
            line = input("teach> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        args = shlex.split(rest) if rest else []
        try:
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("help", "h", "?"):
                print(HELP)
            elif cmd in ("s", "status"):
                q = d.read_fresh() if not d.enabled else np.append(d.q_measured(), d.gripper_measured())
                print(f"  measured  {_fmt(q[:-1])}   gripper {_fmt(q[-1:])}")
                if d.enabled:
                    print(f"  setpoint  {_fmt(d.q_command())}")
                st = asst.status()
                print(f"  enabled={d.enabled} phase={st['phase']} holding={st['holding']} fault={d.fault}")
            elif cmd == "limp":
                d.limp()
                print("  limp - move the arm by hand, then `hold`")
            elif cmd == "hold":
                d.enable()
                print(f"  holding at {_fmt(d.q_command())}")
            elif cmd in ("jog", "set"):
                j, deg = int(args[0]) - 1, float(args[1])
                q = d.q_command()
                q[j] = q[j] + math.radians(deg) if cmd == "jog" else math.radians(deg)
                d.clear_abort()
                d.move_to(q, speed=0.5)
                print(f"  setpoint  {_fmt(d.q_command())}")
            elif cmd == "save":
                name = args[0]
                q = d.q_command() if d.enabled else d.read_fresh()[:-1]
                if np.isnan(q).any():
                    print("  no valid position to save")
                    continue
                d.check_limits(q)
                poses.set(name, q)
                poses.save()
                print(f"  saved {name}: {_fmt(q)}" + ("" if d.enabled else "  (saved while limp; re-save after `hold` for exact replay)"))
            elif cmd == "go":
                d.clear_abort()
                d.move_pose(args[0], speed=float(args[1]) if len(args) > 1 else 1.0)
                print(f"  at {args[0]}")
            elif cmd == "poses":
                for name, p in poses.poses.items():
                    print(f"  {name:12s} {_fmt(p.q)}  {'' if p.taught else '(placeholder)'}")
            elif cmd == "del":
                poses.delete(args[0])
                poses.save()
            elif cmd == "open":
                d.clear_abort()
                d.gripper_open()
            elif cmd == "close":
                d.clear_abort()
                held = d.gripper_grasp()
                print(f"  gripper at {math.degrees(d.gripper_measured()):.1f} deg - {'HOLDING' if held else 'empty'}")
            elif cmd == "g":
                d.set_gripper(math.radians(float(args[0])))
            elif cmd == "pose":
                q = d.q_measured() if d.enabled else d.read_fresh()[:-1]
                xyz = app.kin.fk(q)[:3, 3] * 1000
                print(f"  tool point x={xyz[0]:+.1f} y={xyz[1]:+.1f} z={xyz[2]:+.1f} mm, "
                      f"{math.degrees(tilt_from_down(app.kin, q)):.1f} deg from straight down"
                      + (f", {xyz[2] - app.cfg.grab.table_z * 1000:+.1f} mm above table"
                         if app.cfg.grab.table_z is not None else ""))
            elif cmd == "down":
                d.clear_abort()
                tilt = point_down(d, app.kin)
                print(f"  pointing down ({math.degrees(tilt):.1f} deg off)")
            elif cmd == "table":
                # Works limp (hold the fingertips on the table by hand) or holding (jogged down to touch).
                q = d.q_measured() if d.enabled else d.read_fresh()[:-1]
                z = float(app.kin.fk(q)[2, 3])
                save_table_z(app.cfg, z)
                print(f"  table height = {z * 1000:.1f} mm (tool-point z, written to arm.yaml)")
            elif cmd == "grab":
                d.clear_abort()
                depth = float(args[0]) / 1000 if args else None
                print("  " + str(grab_below(d, app.kin, depth=depth)))
            elif cmd == "gcal":
                which = args[0]
                pos = d.gripper_measured() if d.enabled else float(d.read_fresh()[-1])
                if which not in ("open", "closed") or math.isnan(pos):
                    print("  usage: gcal open|closed (with a valid gripper reading)")
                    continue
                save_gripper_calibration(app.cfg, pos if which == "open" else None, pos if which == "closed" else None)
                print(f"  gripper {which} = {math.degrees(pos):.2f} deg (written to arm.yaml)")
            elif cmd == "check":
                problems = check_poses(app)
                print("\n".join(f"  - {p}" for p in problems) or "  all set")
            elif cmd in ("test", "stage"):
                fields = parse_freeform(rest)
                r = app.catalog.resolve(fields.get("instrument", ""), fields.get("suture"), fields.get("size"),
                                        fields.get("orientation"))
                print(f"  -> slot {r.slot.id}: {r.describe()}")
                asst.submit("handoff" if cmd == "test" else "stage", r)
                print("  running; `release` when the surgeon has it, `stop` to freeze, `s` for status")
            elif cmd == "release":
                print(f"  {asst.release()}")
            elif cmd == "return":
                asst.submit("return")
            elif cmd == "stop":
                asst.stop()
            elif cmd == "reset":
                asst.reset()
            else:
                print("  unknown command; `help`")
        except (IndexError, ValueError) as e:
            if isinstance(e, ResolveError):
                print(f"  {e}")
            else:
                print(f"  bad arguments ({e}); `help`")
        except Busy as e:
            print(f"  {e}")
        except Exception as e:  # noqa: BLE001 - keep the teach session alive
            print(f"  error: {type(e).__name__}: {e}")
    park_and_shutdown(app)


def park_and_shutdown(app: App) -> None:
    d = app.driver
    try:
        if d.enabled and "rest" in app.poses.poses and (app.poses.poses["rest"].taught or d.sim):
            d.clear_fault()
            print("moving to rest...", flush=True)
            d.move_pose("rest", speed=0.5)
    except Exception as e:  # noqa: BLE001
        print(f"could not reach rest pose: {e}", flush=True)
    print("disabling motors", flush=True)
    d.shutdown()
