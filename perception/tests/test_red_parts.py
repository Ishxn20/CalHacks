"""The red-parts identifier on a real scan frame (held out from fitting): all 8 parts, right names."""
import json
from pathlib import Path

import cv2
import numpy as np

from scrub_vision.red_parts import RedPartsDetector

DATA = Path(__file__).parent / "data"


def test_scan_frame_finds_all_eight_parts():
    bgr = cv2.imread(str(DATA / "red_parts_scan.jpg"))
    q = np.array(json.loads((DATA / "red_parts_scan.json").read_text())["q"])
    dets = sorted((d for d in RedPartsDetector().detect(bgr, q) if d.candidates), key=lambda d: -d.grasp_mm[1])
    got = [d.fine_id or ("maybe " + " / ".join(d.candidates)) for d in dets]
    assert got[:7] == ["scalpel_11_long", "needle_holder_ryder", "scissors_micro_forward", "needle_holder_short",
                       "scalpel_10", "forceps_fine_vascular", "scissors_micro_forward"], got
    # the #20 touched the old right finger in this photo; with the re-taped fingers' mask it may count as complete
    assert got[7] in ("scalpel_20", "maybe scalpel_11_long / scalpel_20"), got
    thin = ("scissors_micro_forward", "needle_holder_short", "needle_holder_ryder")  # grabbed at their thinnest point
    assert all(380 < d.grasp_mm[0] < 520 for d in dets if d.fine_id not in thin)
    assert all(350 < d.grasp_mm[0] < 560 for d in dets)
    assert all(2 < d.width_mm < 25 for d in dets)  # across the jaws at the grab point: the arm closes a few mm past it
