"""Watch a surgery video and hand over Potts scissors when the surgeon is about to need them.

    pip install -r predict/requirements.txt
    python -m scrub_arm serve --sim                 # arm server (from arm/), in another terminal
    python predict/watch.py surgery.mp4             # or a camera index, e.g. 1 for an iPhone (Continuity Camera)
    python predict/watch.py surgery.mp4 --dry-run   # predictions only, no arm

Every --every seconds of video, the last --frames frames (spread over --window seconds)
go to an OpenAI vision model, which says whether scissors are needed now. --confirm confident
yeses in a row send POST /handoff to the arm server. Every answer is appended to --log as one JSON line,
with its video time, so a run doubles as a timeline you can review.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import openai

SYSTEM = """\
You are the eyes of a robotic scrub nurse during cardiac or vascular surgery. You see the \
last few frames of the operative video, oldest first. Decide whether the surgeon needs Potts \
(angled) scissors now or within the next few seconds.

Potts scissors are needed right after an artery has been opened with a scalpel blade \
(a small incision in the vessel, the blade being withdrawn), to lengthen that incision.
They are NOT needed if scissors are already in the surgeon's hand, or during suturing, \
dissection, or when no fresh vessel incision is visible.

Describe what you see in one short sentence first, then answer."""

SCHEMA = {
    "type": "object",
    "properties": {
        "seeing": {"type": "string"},
        "needs_scissors": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["seeing", "needs_scissors", "confidence"],
    "additionalProperties": False,
}


def encode(frame, max_side: int = 512) -> str:
    """Shrink (image tokens scale with pixels) and JPEG/base64 a BGR frame."""
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("could not encode frame")
    return base64.standard_b64encode(buf.tobytes()).decode()


def ask(client: openai.OpenAI, model: str, frames: list[tuple[float, str]]) -> dict | None:
    content: list[dict] = []
    for t, jpg in frames:
        content.append({"type": "text", "text": f"t={t:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg}"}})
    content.append({"type": "text", "text": "Does the surgeon need Potts scissors now?"})
    try:
        r = client.chat.completions.create(
            model=model,
            max_completion_tokens=4000,
            reasoning_effort="low",
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA}},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        )
    except (openai.APIConnectionError, openai.APIStatusError) as e:
        print(f"  model call failed: {e}")
        return None
    msg = r.choices[0].message
    if msg.refusal:
        print(f"  model declined these frames: {msg.refusal}")
        return None
    return json.loads(msg.content) if msg.content else None


def is_yes(ans: dict | None, threshold: float) -> bool:
    return bool(ans) and ans["needs_scissors"] and ans["confidence"] >= threshold


def should_fire(streak: int, confirm: int, now: float, last_fire: float | None, cooldown: float) -> bool:
    """`confirm` confident yeses in a row (one shaky frame must not move the arm), and not within
    `cooldown` seconds of the last handoff (one cut needs one pair of scissors)."""
    return streak >= confirm and (last_fire is None or now - last_fire >= cooldown)


def handoff(url: str, instrument: str, orientation: str) -> None:
    body = json.dumps({"instrument": instrument, "orientation": orientation}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/handoff", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  ARM: {json.loads(resp.read()).get('say')}")
    except urllib.error.HTTPError as e:  # 409 busy / 400 not on tray: the server says why
        print(f"  ARM refused: {json.loads(e.read() or b'{}').get('say') or e}")
    except urllib.error.URLError as e:
        print(f"  ARM unreachable: {e.reason}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Video -> OpenAI vision model -> hand over Potts scissors")
    p.add_argument("source", help="video file, or camera index (0, 1, ...)")
    p.add_argument("--model", default="gpt-6.1-sol", help="gpt-6-luna is cheaper and faster")
    p.add_argument("--every", type=float, default=3.0, help="seconds of video between model calls")
    p.add_argument("--frames", type=int, default=5, help="frames per call")
    p.add_argument("--window", type=float, default=15.0, help="seconds of video those frames span")
    p.add_argument("--threshold", type=float, default=0.85, help="minimum confidence for a yes")
    p.add_argument("--confirm", type=int, default=2, help="consecutive yeses needed to hand over")
    p.add_argument("--cooldown", type=float, default=30.0, help="seconds before another handoff")
    p.add_argument("--instrument", default="potts scissors")
    p.add_argument("--orientation", default="forehand")
    p.add_argument("--arm-url", default="http://127.0.0.1:8765")
    p.add_argument("--log", default="predictions.jsonl")
    p.add_argument("--dry-run", action="store_true", help="print predictions, don't move the arm")
    args = p.parse_args(argv)

    live = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if live else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.source}")
    delay_ms = 1 if live else max(1, int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)))

    client = openai.OpenAI()  # reads OPENAI_API_KEY
    pool = ThreadPoolExecutor(max_workers=1)  # one call in flight; playback never waits on the model
    pending: tuple[float, Future] | None = None
    buf: deque[tuple[float, str]] = deque(maxlen=args.frames)
    gap = args.window / args.frames
    last_sample = last_ask = last_fire = None
    streak = 0
    label = "watching..."
    t0 = time.monotonic()

    with open(args.log, "a") as log:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.monotonic() - t0 if live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            if last_sample is None or t - last_sample >= gap:
                buf.append((t, encode(frame)))
                last_sample = t
            if pending is None and len(buf) == args.frames and (last_ask is None or t - last_ask >= args.every):
                pending = (t, pool.submit(ask, client, args.model, list(buf)))
                last_ask = t
            if pending is not None and pending[1].done():
                asked_at, fut = pending
                pending = None
                ans = fut.result()
                if ans:
                    label = f"{'SCISSORS' if ans['needs_scissors'] else 'no'} {ans['confidence']:.2f} - {ans['seeing']}"
                    print(f"[{asked_at:6.1f}s] {label}")
                    log.write(json.dumps({"t": round(asked_at, 1), **ans}) + "\n")
                    log.flush()
                    streak = streak + 1 if is_yes(ans, args.threshold) else 0
                    if should_fire(streak, args.confirm, asked_at, last_fire, args.cooldown):
                        last_fire, streak = asked_at, 0  # the next handoff needs fresh confirmations
                        print(f"  -> handing over {args.instrument}")
                        if not args.dry_run:
                            handoff(args.arm_url, args.instrument, args.orientation)

            cv2.putText(frame, label[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("watch (q to quit)", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
