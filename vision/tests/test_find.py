import json

import numpy as np

from scrub_vision import find

BOX = (10, 10, 50, 40)


def test_requested_class_wins_even_if_another_class_is_more_confident():
    # #11 (class 8) scores 0.95, #20 (class 10) scores 0.6: asking for #20 must return the #20.
    dets = [(8, 0.95, BOX), (10, 0.60, (60, 10, 100, 40))]
    assert find.select(dets, 10, 0.5)[1] == 0.60


def test_best_of_same_class_and_threshold():
    dets = [(10, 0.55, BOX), (10, 0.80, (60, 10, 100, 40)), (10, 0.30, (1, 1, 5, 5))]
    assert find.select(dets, 10, 0.5)[1] == 0.80
    assert find.select(dets, 10, 0.9) is None


def test_absent_class_is_not_found():
    assert find.select([(8, 0.99, BOX)], 10, 0.5) is None


def test_write_result_roundtrip_and_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(find, "REPO", tmp_path)
    monkeypatch.setattr(find, "OUT_JSON", tmp_path / "d.json")
    monkeypatch.setattr(find, "OUT_CROP", tmp_path / "c.jpg")
    monkeypatch.setattr(find, "OUT_ANN", tmp_path / "a.jpg")
    frame = np.full((100, 120, 3), 90, np.uint8)
    out = find.write_result("scalpel_20", (10, 0.9, BOX), frame, heard="hand me a scalpel 20")
    assert json.loads((tmp_path / "d.json").read_text()) == out
    assert out["found"] and out["bbox_xyxy"] == [10, 10, 50, 40]
    assert (tmp_path / "c.jpg").exists() and (tmp_path / "a.jpg").exists()
    miss = find.write_result("scalpel_20", None, frame, error="not found")
    assert miss["found"] is False and miss["bbox_xyxy"] is None
