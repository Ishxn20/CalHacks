"""Goal check: is every tool in the set found and identified at the scan pose?

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/check_ids.py            (scan pose, live camera)
    .venv/bin/python tools/check_ids.py --frame hw_check/real/afterfault/scan_1792911745_1   (a saved <stem>.jpg/.json)

Prints every red part with its identity (or the candidates it can't choose between) and exits 0 only if each tool in
tray.yaml `in_set` is found whole with a sure identity (the two micro scissors count as one). Add --expect to check
a different set: --expect "10 blade" tweezers ...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

R = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
sys.path.insert(0, R + "/perception")
from scrub_arm.catalog import Catalog                               # noqa: E402
from scrub_arm.config import load_tray_config                       # noqa: E402
from scrub_vision.red_parts import SAME_PART, RedPartsDetector   # noqa: E402


def frame_from_arm():
    from scrub_arm.app import build
    from scrub_arm.teach import park_and_shutdown
    app = build()
    d = app.driver
    d.start(); d.enable()
    try:
        d.move_pose("scan", speed=0.5); d.gripper_open()
        cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
        for _ in range(15):
            cap.read()
        ok, img = cap.read()
        q = d.q_measured()
        cap.release()
        if not ok:
            sys.exit("the wrist camera gave no picture")
        out = Path(R) / "hw_check/real/check_ids"; out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "scan.jpg"), img); (out / "scan.json").write_text(json.dumps({"q": q.tolist()}))
        return img, q
    finally:
        park_and_shutdown(app)


def main() -> None:
    args = sys.argv[1:]
    tray = load_tray_config()
    cat = Catalog(tray, require_slot=False)
    expect = [cat.resolve(n).instrument.id for n in args[args.index("--expect") + 1:]] if "--expect" in args \
        else list(tray.in_set)
    if "--frame" in args:
        stem = args[args.index("--frame") + 1]
        img, q = cv2.imread(stem + ".jpg"), np.array(json.load(open(stem + ".json"))["q"])
    else:
        img, q = frame_from_arm()
    dets = RedPartsDetector().detect(img, q)
    group_of = {i: ids[0] for ids in SAME_PART.values() for i in ids}   # interchangeable ids (the micro scissors)
    groups = {}
    for fid in expect:
        groups.setdefault(group_of.get(fid, fid), set()).add(fid)
    sure = set()
    for x in sorted(dets, key=lambda d: -d.grasp_mm[1]):
        if not x.candidates:
            print(f"  ?  {x.length_mm:5.0f} mm long at ({x.grasp_mm[0]:.0f}, {x.grasp_mm[1]:.0f}): not one of ours")
            continue
        name = x.fine_id or "could be " + " / ".join(x.candidates)
        print(f"  {'ok' if x.fine_id and x.exposure == 1 else '--':2s} {name:40s} at ({x.grasp_mm[0]:.0f}, {x.grasp_mm[1]:.0f})"
              f"  {x.length_mm:.0f} mm  {'whole' if x.exposure == 1 else f'{x.exposure:.0%} in view'}")
        if x.fine_id and x.exposure == 1:
            sure.add(group_of.get(x.fine_id, x.fine_id))
    missing = [sorted(v)[0] if len(v) == 1 else "micro scissors" for k, v in groups.items() if k not in sure]
    print(f"\n{len(sure)}/{len(groups)} identified" + (f"; not identified: {', '.join(missing)}" if missing else ": all parts identified"))
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
