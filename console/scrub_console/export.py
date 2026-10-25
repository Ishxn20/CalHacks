"""Export accepted episodes as a world-model training set.

    python -m scrub_console export --site-url $CONVEX_SITE_URL --key $EXPORT_KEY --out datasets/scrub_v1

Layout (LeRobot-style: one row per video frame, observation/action aligned by index):

    meta/info.json            fps, feature shapes, episode count, joint names
    meta/episodes.jsonl       per episode: task text, caption, success, failure mode, segments
    data/episode_XXXXXX.npz   observation_state [T,7] (6 joints + gripper, deg),
                              action [T,7] (commanded joints + gripper, deg), timestamp [T] (s), phase [T]
    videos/episode_XXXXXX.mp4 de-identified wrist camera

Only episodes Convex accepted are exported: consented, labelled, and with no privacy flags.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def _get(url: str, headers: dict | None = None) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as r:
        return r.read()


def fetch_manifest(site_url: str, key: str) -> list[dict]:
    return json.loads(_get(site_url.rstrip("/") + "/export/manifest", {"Authorization": f"Bearer {key}"}))


def search_ids(convex_url: str, text: str, client=None) -> set[str]:
    """Ids of accepted episodes whose Claude labels match `text` (Convex full-text search, episodes:search)."""
    if client is None:
        from convex import ConvexClient

        client = ConvexClient(convex_url)
    return {e["id"] for e in client.query("episodes:search", {"q": text, "acceptedOnly": True})}


def frames_to_arrays(lines: list[dict]) -> dict[str, np.ndarray]:
    def vec(q: list, g) -> list[float]:
        q = list(q or []) + [None] * (6 - len(q or []))
        return [np.nan if v is None else float(v) for v in q[:6]] + [np.nan if g is None else float(g)]

    t0 = lines[0]["t"] if lines else 0.0
    return {
        "observation_state": np.array([vec(f.get("qDeg"), f.get("gripperDeg")) for f in lines], np.float32).reshape(-1, 7),
        "action": np.array([vec(f.get("qCmdDeg"), f.get("gripperCmdDeg")) for f in lines], np.float32).reshape(-1, 7),
        "timestamp": np.array([f["t"] - t0 for f in lines], np.float64),
        "phase": np.array([f.get("phase", "unknown") for f in lines]),
    }


def export(manifest: list[dict], out: Path, fetch=_get) -> int:
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    rows, fps = [], set()
    for idx, ep in enumerate(sorted(manifest, key=lambda e: e["startedAt"])):
        lines = [json.loads(line) for line in fetch(ep["framesUrl"]).decode().splitlines() if line.strip()]
        if not lines:
            continue
        np.savez_compressed(out / "data" / f"episode_{idx:06d}.npz", **frames_to_arrays(lines))
        (out / "videos" / f"episode_{idx:06d}.mp4").write_bytes(fetch(ep["videoUrl"]))
        lab = ep["labels"]
        rows.append({"episode_index": idx, "source_id": ep["id"], "length": len(lines),
                     "task": lab["task"], "request": ep["request"], "caption": lab["caption"],
                     "success": lab["success"], "failure_mode": lab["failureMode"], "segments": lab["segments"],
                     "labeler": lab["model"]})
        fps.add(ep["fps"])
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "meta" / "info.json").write_text(json.dumps({
        "codebase": "scrub_console", "robot_type": "seeed_rebot_b601_rs", "fps": sorted(fps),
        "total_episodes": len(rows), "total_frames": sum(r["length"] for r in rows),
        "features": {"observation_state": {"shape": [7], "names": JOINTS, "unit": "deg"},
                     "action": {"shape": [7], "names": JOINTS, "unit": "deg"},
                     "video": {"camera": "wrist", "deidentified": True}},
    }, indent=2))
    return len(rows)
