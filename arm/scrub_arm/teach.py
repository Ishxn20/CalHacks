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
from .config import save_gripper_calibration
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
  g <deg>                 gripper to an absolute motor angle
  pose                    tool-point position (mm) and how far the gripper is from pointing straight down
  ik <x> <y> <z> [yaw]    move to a top-down tool pose (mm / deg), e.g. ik 220 0 140 (a scan candidate)
  pivot [yaw]             turn the gripper to point straight down about its tip (tip stays put);
                          yaw = jaw direction in deg (default: keep the current jaw heading)
  grabhere [thick] [yaw]  pick up what's under the tip: pivot down, open, descend to contact, grasp, lift
                          (thick = instrument thickness in mm, default 3)
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
    _kin = []

    def kin():
        if not _kin:
            from .kinematics import Kinematics   # pinocchio; only needed for `pose` / `ik`

            _kin.append(asst.kin or Kinematics())
        return _kin[0]

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
                p, R = kin().fk(q)
                tilt = math.degrees(math.acos(np.clip(-(R @ kin().tool_axis_local)[2], -1, 1)))
                print(f"  tool point x={p[0]*1000:+.1f} y={p[1]*1000:+.1f} z={p[2]*1000:+.1f} mm, "
                      f"{tilt:.1f} deg from straight down")
            elif cmd == "ik":
                x, y, z = (float(a) / 1000 for a in args[:3])
                yaw = math.radians(float(args[3])) if len(args) > 3 else 0.0
                q = kin().ik_topdown(x, y, z, yaw, d.q_command(), (app.cfg.lower, app.cfg.upper),
                                                 allow_flip=False)
                if q is None:
                    print("  unreachable top-down")
                    continue
                print(f"  moving to {_fmt(q)}")
                d.clear_abort()
                d.move_to(q, speed=0.5)
            elif cmd in ("pivot", "grabhere"):
                if asst.kin is None:
                    asst.kin = kin()
                d.clear_abort()
                if cmd == "pivot":
                    x, y, z, yaw = asst.pivot_down(float(args[0]) if args else None)
                    print(f"  pointing down at x={x:.0f} y={y:.0f} mm, tip {z * 1000:.0f} mm up, jaws at {yaw:.0f} deg "
                          f"({math.degrees(kin().tilt_from_down(d.q_command())):.1f} deg off vertical)")
                else:
                    thick = float(args[0]) if args else 3.0
                    asst.grab_here(thick, float(args[1]) if len(args) > 1 else None)
                    print("  HOLDING" if d.gripper_holding() else "  empty")
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
                print(f"  -> slot {r.slot.id if r.slot else 'camera'}: {r.describe()}")
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
