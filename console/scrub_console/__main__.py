"""python -m scrub_console {demo,serve,say,check,export}

    demo [--fast] [--agents]                     simulated arm + console + dashboard in one process, no hardware
    serve [--sim] [--demo] [--rehearse-droid] [--agents] [--record CAMERA|sim]
                                                 the dashboard at http://127.0.0.1:8780 (arm server must be running)
    export --site-url URL --key KEY --out DIR    accepted episodes as a LeRobot-style training set
    say "needle holder, 6-0 prolene, backhand"   one request through the same path the dashboard uses
    check                                        which integrations are configured, and is the arm reachable
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path

from .app import build_console
from .arm_client import ArmError
from .config import Settings


def _banner(caps: dict) -> None:
    """Which integrations are really live. A rehearsal Droid is not an agent, so it is listed apart."""
    flags = {k: v for k, v in caps.items() if k not in ("sim", "demo", "convex_url", "agents")}
    live = [k for k, v in flags.items() if v and v != "rehearsal"]
    off = [k for k, v in flags.items() if not v]
    print("  live:    " + (", ".join(live) or "rules only"), flush=True)
    if flags.get("droid") == "rehearsal":
        print("  rehearsal only (no agent runs): droid", flush=True)
    print("  not set: " + (", ".join(off) or "nothing"), flush=True)


def cmd_serve(args) -> None:
    from .server import serve

    c = build_console(Settings.from_env(), sim=args.sim, demo=args.demo, rehearse_droid=args.rehearse_droid, agents=args.agents,
                      record=args.record)
    stop = threading.Event()
    threading.Thread(target=c.bridge.run, args=(args.poll, stop), name="poll", daemon=True).start()
    httpd = serve(c, args.host, args.port)
    caps = c.capabilities()
    print(f"scrub_console on http://{args.host}:{args.port}  "
          f"({'SIMULATED arm' if args.sim else 'real arm'}{', demo tools on' if args.demo else ''})", flush=True)
    if c.agents:
        print("  agents:  " + ", ".join(f"{k}={v}" for k, v in c.agents.addresses.items()), flush=True)
    _banner(caps)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        stop.set()
        httpd.server_close()


def cmd_demo(args) -> None:
    from .demo import start

    d = start(fast=args.fast, agents=args.agents, port=args.port)
    caps = d.console.capabilities()
    print(f"scrub_console rehearsal on {d.url}  (SIMULATED arm; files in {d.work})", flush=True)
    _banner(caps)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        d.stop()


def cmd_export(args) -> None:
    from .export import fetch_manifest, export

    if not args.site_url or not args.key:
        raise SystemExit("--site-url and --key are required (or CONVEX_SITE_URL / EXPORT_KEY)")
    n = export(fetch_manifest(args.site_url, args.key), Path(args.out))
    print(f"exported {n} episodes to {args.out}")


def cmd_say(args) -> None:
    c = build_console(Settings.from_env(), sim=args.sim)
    print(json.dumps(c.bridge.handle_text(" ".join(args.text), via="cli"), indent=2))


def cmd_check(args) -> None:
    c = build_console(Settings.from_env(), sim=args.sim)
    print(json.dumps(c.capabilities(), indent=2))
    try:
        st = c.arm.status()
        print(f"arm reachable at {c.settings.arm_url}: phase={st.get('phase')}")
        print("arm audit:", json.dumps(c.arm.audit()))
    except ArmError as e:
        print(f"arm NOT reachable at {c.settings.arm_url}: {e}")
        raise SystemExit(1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="scrub_console", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8780)
    s.add_argument("--poll", type=float, default=0.4, help="seconds between arm status polls")
    s.add_argument("--sim", action="store_true", help="the arm is simulated: enables the simulated count and sim controls")
    s.add_argument("--demo", action="store_true", help="enable the fault-injection and audit-tamper demo buttons")
    s.add_argument("--rehearse-droid", action="store_true",
                   help="if the Droid CLI is not installed, rehearse triage with a canned (clearly labelled) edit")
    s.add_argument("--agents", action="store_true",
                   help="run voice / arm / eyes as Fetch.ai uAgents that carry the commands (needs the agents extra)")
    s.add_argument("--record", metavar="CAMERA|sim",
                   help="record every arm job for the training set: a wrist-camera index, or 'sim' for placeholder "
                        "frames (needs Convex and the dataset extra; turning it on is consent for research use)")
    s.set_defaults(fn=cmd_serve)
    x = sub.add_parser("export", help="download accepted episodes as a LeRobot-style training set")
    x.add_argument("--site-url", default=os.environ.get("CONVEX_SITE_URL"), help="Convex HTTP actions URL")
    x.add_argument("--key", default=os.environ.get("EXPORT_KEY"))
    x.add_argument("--out", default="datasets/scrub_v1")
    x.set_defaults(fn=cmd_export)
    r = sub.add_parser("demo", help="simulated arm + console + dashboard in one process (no hardware)")
    r.add_argument("--port", type=int, default=8780)
    r.add_argument("--fast", action="store_true", help="speed the simulated arm up")
    r.add_argument("--agents", action="store_true", help="run voice / arm / eyes as Fetch.ai uAgents")
    r.set_defaults(fn=cmd_demo)
    t = sub.add_parser("say")
    t.add_argument("text", nargs="+")
    t.add_argument("--sim", action="store_true")
    t.set_defaults(fn=cmd_say)
    c = sub.add_parser("check")
    c.add_argument("--sim", action="store_true")
    c.set_defaults(fn=cmd_check)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
