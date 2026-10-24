import json
from types import SimpleNamespace

import numpy as np

from scrub_vision import train

NAMES = ["a", "b", "c"]


def test_confusions_finds_most_confused_other_class():
    # matrix[pred, true]; last row/col is background. true b: 6 right, 3 called c, 1 missed.
    m = np.zeros((4, 4))
    m[0, 0] = 10
    m[1, 1], m[2, 1], m[3, 1] = 6, 3, 1
    m[2, 2] = 5
    rows = {r["true"]: r for r in train.confusions(m, NAMES)}
    assert rows["b"] == {"true": "b", "pred": "c", "rate": 0.3}
    assert rows["a"]["rate"] == 0.0


def test_export_results_json(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    run = tmp_path / "runs" / "r"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    (run / "results.csv").write_text("epoch\n1\n2\n")
    for s in ("train", "val"):
        (tmp_path / "data" / "clean" / "images" / s).mkdir(parents=True)
    (tmp_path / "data" / "clean" / "images" / "train" / "x.jpg").write_bytes(b"")
    box = SimpleNamespace(map50=0.9, map=0.7, p=[1, 1, 1], r=[1, 1, 1], ap50=[1, 1, 1])
    args = SimpleNamespace(data=tmp_path / "data" / "clean" / "data.yaml", model="m.pt", imgsz=64, batch=2,
                           epochs=2, max_confusion=0.05)
    rows = [{"true": "a", "pred": "b", "rate": 0.2}]
    out = train.export_results("r", args, SimpleNamespace(box=box), NAMES, rows, 5.0)
    rep = json.loads((out / "results.json").read_text())
    assert rep["epochs_completed"] == 2 and rep["dataset"] == {"train": 1, "val": 0}
    assert rep["confusion_pass"] is False and rep["confusion_failures"] == rows
    assert (out / "best.pt").exists() and (out / "results.csv").exists()
