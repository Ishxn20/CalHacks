import pytest
import yaml

from scrub_vision.synth import make_dataset
from scrub_vision.train import combine, train


def test_combine_lists_every_train_dir(tmp_path):
    a = make_dataset(tmp_path / "a", n=2, seed=0, val_frac=0.5)
    b = make_dataset(tmp_path / "b", n=2, seed=1, val_frac=0.5)
    out = combine([a, b], tmp_path / "mixed.yaml")
    d = yaml.safe_load(out.read_text())
    assert len(d["train"]) == 2 and len(d["val"]) == 2
    assert d["names"] == yaml.safe_load(a.read_text())["names"]


@pytest.mark.slow
def test_one_epoch_produces_weights(tmp_path):
    data = make_dataset(tmp_path / "d", n=8, seed=0, val_frac=0.25)
    best = train([data], epochs=1, imgsz=320, project=tmp_path / "runs", name="smoke", batch=4)
    assert best.exists()
