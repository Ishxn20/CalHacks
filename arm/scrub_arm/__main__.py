"""python -m scrub_arm {serve,teach,check,send} [--sim]"""
from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import urllib.error
import urllib.request

from .app import build, check_poses


def _raise_interrupt(*_) -> None:
    raise KeyboardInterrupt


def cmd_serve(args) -> None:
    from .server import serve
    from .teach import park_and_shutdown

    app = build(sim=args.sim, config_dir=args.config_dir)
    problems = check_poses(app)
    if problems and not args.sim:
        print("Setup problems (fix with `python -m scrub_arm teach`):\n" + "\n".join(f"  - {p}" for p in problems))
        if not args.force:
            sys.exit(1)
    app.driver.start()
    app.driver.enable()
    if args.home:
        app.driver.move_pose("home")
    host, port = args.host or app.cfg.server_host, args.port or app.cfg.server_port
    httpd = serve(app.assistant, app.catalog, host, port)
    signal.signal(signal.SIGTERM, _raise_interrupt)  # treat `kill` like Ctrl-C so the arm gets parked
    print(f"scrub_arm {'SIM ' if args.sim else ''}listening on http://{host}:{port}  (Ctrl-C to quit)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        httpd.server_close()
        app.assistant.stop()
        try:
            app.assistant.wait_idle(timeout=5.0)
        except TimeoutError:
            pass
        park_and_shutdown(app)


def cmd_teach(args) -> None:
    from .teach import run_teach

    run_teach(build(sim=args.sim, config_dir=args.config_dir))


def cmd_check(args) -> None:
    app = build(sim=args.sim, config_dir=args.config_dir)
    app.driver.bus.connect()
    ok = True
    for name, err in app.driver.bus.ping().items():
        print(f"  {name:8s} {'OK' if not err else 'NO RESPONSE: ' + err}")
        ok &= err is None
    q = app.driver.bus.read_fresh()
    print("  positions (deg): " + " ".join(f"{math.degrees(x):+.1f}" for x in q))
    problems = check_poses(app)
    print("Setup:\n" + ("\n".join(f"  - {p}" for p in problems) or "  all set"))
    app.driver.bus.close()
    sys.exit(0 if ok else 1)


def cmd_send(args) -> None:
    url = args.url.rstrip("/") + "/" + args.action
    if args.action == "status":
        req = urllib.request.Request(url)
    else:
        body = {"text": " ".join(args.text)} if args.text else {}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(json.dumps(json.loads(resp.read()), indent=2))
    except urllib.error.HTTPError as e:
        print(json.dumps(json.loads(e.read() or b"{}"), indent=2))
        sys.exit(1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="scrub_arm", description="Robotic scrub assistant - arm control")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--sim", action="store_true", help="run against the simulator, no hardware")
        sp.add_argument("--config-dir", default=None)

    s = sub.add_parser("serve", help="run the HTTP API the voice pipeline talks to")
    common(s)
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--home", action="store_true", help="move to the home pose on startup")
    s.add_argument("--force", action="store_true", help="start even if poses/gripper aren't set up")
    s.set_defaults(fn=cmd_serve)

    t = sub.add_parser("teach", help="interactive pose teaching + manual tests")
    common(t)
    t.set_defaults(fn=cmd_teach)

    c = sub.add_parser("check", help="ping all 7 motors, print positions and setup problems")
    common(c)
    c.set_defaults(fn=cmd_check)

    snd = sub.add_parser("send", help="send a command to a running server")
    snd.add_argument("action", choices=["handoff", "stage", "resolve", "release", "return", "home", "stop", "reset", "status"])
    snd.add_argument("text", nargs="*", help='e.g. "needle holder, 6-0 prolene, backhand"')
    snd.add_argument("--url", default="http://127.0.0.1:8765")
    snd.set_defaults(fn=cmd_send)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
