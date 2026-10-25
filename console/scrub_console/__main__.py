"""python -m scrub_console {serve,say,check}

    serve [--sim] [--demo] [--rehearse-droid]   the dashboard at http://127.0.0.1:8780 (arm server must be running)
    say "needle holder, 6-0 prolene, backhand"   one request through the same path the dashboard uses
    check                                        which integrations are configured, and is the arm reachable
"""
from __future__ import annotations

import argparse
import json
import logging
import threading

from .app import build_console
from .arm_client import ArmError
from .config import Settings


def cmd_serve(args) -> None:
    from .server import serve

    c = build_console(Settings.from_env(), sim=args.sim, demo=args.demo, rehearse_droid=args.rehearse_droid)
    stop = threading.Event()
    threading.Thread(target=c.bridge.run, args=(args.poll, stop), name="poll", daemon=True).start()
    httpd = serve(c, args.host, args.port)
    caps = c.capabilities()
    print(f"scrub_console on http://{args.host}:{args.port}  "
          f"({'SIMULATED arm' if args.sim else 'real arm'}{', demo tools on' if args.demo else ''})", flush=True)
    flags = {k: v for k, v in caps.items() if k not in ("sim", "demo", "convex_url")}
    print("  live:    " + (", ".join(k for k, v in flags.items() if v) or "rules only"), flush=True)
    print("  not set: " + (", ".join(k for k, v in flags.items() if not v) or "nothing"), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        stop.set()
        httpd.server_close()


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
    s.set_defaults(fn=cmd_serve)
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
