"""Client for the tray-camera service (`python -m scrub_vision.serve`, owned by the vision side).

    POST /detect  {"names": ["scalpel_11"] | null, "frames": 5}
      -> {"found": bool, "name", "conf", "center_px": [u, v], "grip_deg", "long_deg", "length_px", "width_px",
          "heavy_dir": [dx, dy], "box_px": [[u, v] x4], "stable": bool, "spread_px", "spread_deg",
          "identity": "yolo" | "unchecked", "others": [{"name", "box_px"}], "markers": {"<id>": [u, v]},
          "image": [w, h]}
    GET  /markers -> {"markers": {"<id>": {"center": [u, v], "side_px": float}}, "image": [w, h]}

Image angles: x right, y down, positive clockwise on screen, in [-90, 90). `grip_deg` is the direction
the jaws must close along.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import PickCfg


class VisionError(RuntimeError):
    pass


class VisionClient:
    def __init__(self, cfg: PickCfg) -> None:
        self.cfg = cfg

    def _call(self, path: str, body: dict | None = None) -> dict:
        if not self.cfg.vision_url:
            raise VisionError("pick.vision_url is not set in arm.yaml")
        req = urllib.request.Request(
            self.cfg.vision_url.rstrip("/") + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise VisionError(f"tray camera service unreachable ({e})") from e

    def detect(self, names: list[str] | None, frames: int | None = None) -> dict:
        return self._call("/detect", {"names": names, "frames": frames or self.cfg.frames})

    def markers(self) -> dict:
        return self._call("/markers")
