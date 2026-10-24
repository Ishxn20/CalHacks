# scrub_vision: find the requested instrument in the tray image

Reads `last_command.json` (from `voice/`), finds that instrument in the camera image, and writes the crop plus a JSON result. The detector is trained on **every** instrument in `config/tray.yaml`, loaded or not, so a #11 next to a #20 is never mistaken for it. At run time only detections of the requested class are considered.

```bash
cd vision
uv venv --python 3.12 && uv pip install -e ../arm -e ".[dev]"
# STLs are tracked in assets/stl/
.venv/bin/python -m scrub_vision.render_dataset --train 4000 --val 600   # synthetic YOLO set, one class per tray.yaml id
.venv/bin/python -m scrub_vision.check_data                              # validate, writes data/clean + rejected.csv
.venv/bin/python -m scrub_vision.train --model yolo11n.pt --imgsz 512 --epochs 60
.venv/bin/python -m scrub_vision.find                                    # live camera, watches ../last_command.json
.venv/bin/python -m scrub_vision.find --image tray.jpg --instrument scalpel_20   # still image
.venv/bin/python -m pytest
```

Outputs at the repo root (gitignored): `last_detection.json`, `last_crop.jpg`, `last_annotated.jpg`.

```json
{"found": true, "instrument": "scalpel_20", "heard": "hand me a scalpel 20", "confidence": 0.91,
 "bbox_xyxy": [120, 80, 310, 150], "crop": "last_crop.jpg", "annotated": "last_annotated.jpg", "error": null, "ts": 0}
```

`found` is false (with `error`) if nothing of that class clears `--min-conf` (default 0.5). Live mode searches camera frames for `--window` seconds (default 2) and keeps the most confident hit, so a slow camera just means fewer frames.

Notes
- `check_data` refuses a dataset whose class list differs from `config/tray.yaml` (same names, same order).
- The coronary probe has no STL, so `render_dataset.py` draws a stand-in rod; replace it when the real part exists.
- Training is on synthetic turquoise renders and intentionally not early-stopped. Check accuracy on real photos of the printed tools before trusting it.
- `train` ends with per-class precision/recall and a confusion report; it flags any tool confused with another above 5%.
