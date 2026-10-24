import cv2
import numpy as np

from scrub_vision import sandbox
from test_grasp import scene


def test_build_labels_one_blob_per_photo_and_skips_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "DATA", tmp_path)
    monkeypatch.setattr(sandbox, "RAW", tmp_path / "raw")
    for name, angle in (("sandbox_a", 0), ("sandbox_b", 50)):
        d = tmp_path / "raw" / name
        d.mkdir(parents=True)
        for i in range(8):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), scene(angle + i))
    cv2.imwrite(str(tmp_path / "raw" / "sandbox_a" / "9999.jpg"), np.full((480, 640, 3), 90, np.uint8))  # empty

    assert sandbox.build() == {"sandbox_a": 8, "sandbox_b": 8}
    yaml = (tmp_path / "data.yaml").read_text()
    assert "0: sandbox_a" in yaml and "1: sandbox_b" in yaml
    label = next((tmp_path / "labels" / "train").glob("sandbox_b_*.txt")).read_text().split()
    assert label[0] == "1" and abs(float(label[1]) - 0.5) < 0.01     # class id, box centred in the frame
    assert list((tmp_path / "images" / "val").glob("*.jpg"))
