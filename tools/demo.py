"""The voice demo in one place. Every subcommand loads the repo-root .env first (SCRUB_API_TOKEN and the service keys).

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/demo.py arm   # motors ON + wrist camera: operator only
    .venv/bin/python tools/demo.py console                                # dashboard on http://127.0.0.1:8780
    .venv/bin/python tools/demo.py check                                  # what's configured, is the arm up (free)
    .venv/bin/python tools/demo.py check --live                           # one real call to each service (paid)
    .venv/bin/python tools/demo.py send stop|release|return|reset|status  # operator commands to the arm
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("arm", "perception", "console")]
from scrub_console.config import Settings, load_dotenv  # noqa: E402


def check(live: bool) -> int:
    from dataclasses import replace

    from scrub_arm.config import load_tray_config
    from scrub_console.arm_client import ArmClient, ArmError

    s, tray = Settings.from_env(), load_tray_config()
    print("demo tools: " + ", ".join(tray.instruments[k].name for k in tray.in_set))
    try:
        print(f"arm: reachable, {ArmClient(s.arm_url, s.arm_token).status()['phase']}")
    except ArmError as e:
        print(f"arm: not reachable ({e})")
    demo_tray = replace(tray, instruments={k: tray.instruments[k] for k in tray.in_set})
    services = {"Claude": (bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")), lambda: _claude(s, demo_tray)),
                "Cartesia": (bool(os.getenv("CARTESIA_API_KEY")), lambda: _cartesia(s)),
                "Convex": (bool(s.convex_url and s.convex_deploy_key), lambda: _convex(s)),
                "Bitdeer": (bool(s.bitdeer_key), lambda: _bitdeer(s))}
    failed = False
    for name, (configured, call) in services.items():
        if not configured:
            failed = True
            print(f"{name}: missing configuration")
        elif not live:
            print(f"{name}: configured")
        else:
            try:
                print(f"{name}: {call()}", flush=True)
            except Exception as e:  # noqa: BLE001 - report every service, whichever fails
                failed = True
                print(f"{name}: FAILED ({type(e).__name__}: {e})", flush=True)
    return int(failed)


def _claude(s, tray) -> str:
    from scrub_console.intent import ClaudeParser
    picked = ClaudeParser(tray, model=s.claude_model).call("hand me the 10 blade", "phase=idle")
    if not picked or picked[0] != "request_instrument":
        raise RuntimeError(f"expected a tool request, got {picked}")
    return f"live intent ok ({picked[1]})"


def _cartesia(s) -> str:
    from scrub_console.speech import CartesiaSpeech
    speech = CartesiaSpeech(api_key=os.getenv("CARTESIA_API_KEY"), voice_id=s.cartesia_voice,
                            tts_model=s.cartesia_tts_model, stt_model=s.cartesia_stt_model)
    text = speech.transcribe(speech.synthesize("Scrub, hand me the ten blade."))
    if not text.strip():
        raise RuntimeError("empty transcription")
    return f"speech out and back in ok ({text!r})"


def _convex(s) -> str:
    from scrub_console.sinks import ConvexHTTP
    tag = uuid.uuid4().hex
    ConvexHTTP(s.convex_url, s.convex_deploy_key).mutation(
        "events:add", {"kind": "service_check", "data": {"check_id": tag}, "seq": 0, "t": time.time()})
    req = urllib.request.Request(s.convex_url.rstrip("/") + "/api/query", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"path": "events:recent", "args": {}, "format": "json"}).encode())
    with urllib.request.urlopen(req, timeout=15) as resp:
        rows = json.loads(resp.read()).get("value") or []
    if not any((r.get("data") or {}).get("check_id") == tag for r in rows):
        raise RuntimeError("the write is not visible to the dashboard's public query")
    return "write and dashboard read ok (one service_check event added)"


def _bitdeer(s) -> str:
    from scrub_console.report import BitdeerClient
    text = BitdeerClient(s.bitdeer_key, s.bitdeer_base_url, s.bitdeer_model).chat(
        [{"role": "user", "content": "Reply with the single word ready."}])
    return f"live inference ok ({text[:40]!r})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("service", choices=["arm", "console", "check", "send"])
    p.add_argument("rest", nargs="*", help="send: the arm command (and text); arm/console: extra flags")
    p.add_argument("--live", action="store_true", help="check: make one real (paid) call to each service")
    args, extra = p.parse_known_args()
    load_dotenv()
    if args.service == "arm":
        from scrub_arm.__main__ import main as arm_main
        arm_main(["serve", *args.rest, *extra])
    elif args.service == "console":
        from scrub_console.__main__ import main as console_main
        console_main(["serve", *args.rest, *extra])
    elif args.service == "send":
        if not args.rest:
            p.error("send needs an arm command, e.g. send stop")
        from scrub_arm.__main__ import main as arm_main
        arm_main(["send", *args.rest])
    else:
        return check(args.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
