import numpy as np

from scrub_vision.render import scene_mesh
from scrub_vision.templates import load_instruments
from sim.pile_sim import poses_source, settle_pile


def world_vertices(mesh, T):
    return mesh.vertices @ T[:3, :3].T + T[:3, 3]


def test_settled_bodies_rest_on_or_above_table():
    insts = load_instruments()
    pile, _ = settle_pile(["forceps_resano", "needle_holder_ryder", "scissors_potts", "clamp_side_biting"],
                          np.random.default_rng(0))
    for fid, T in pile:
        z = world_vertices(scene_mesh(insts[fid]), T)[:, 2]
        assert z.min() > -1.5, (fid, z.min())
        assert z.max() < 120


def test_settled_pose_stops_changing():
    for seed in range(4):
        _, drift = settle_pile(["forceps_resano", "scissors_potts", "forceps_debakey", "needle_holder_ryder",
                                "clamp_side_biting"], np.random.default_rng(seed))
        assert drift < 0.5, (seed, drift)


def test_something_ends_up_stacked():
    insts = load_instruments()
    stacked = 0
    for seed in range(3):
        pile, _ = settle_pile(["clamp_side_biting", "clamp_aortic_cross", "needle_holder_long", "forceps_resano",
                               "needle_holder_ryder", "scissors_potts"], np.random.default_rng(seed))
        stacked += sum(world_vertices(scene_mesh(insts[f]), T)[:, 2].min() > 2.0 for f, T in pile)
    assert stacked > 0


def test_dataset_from_mujoco(tmp_path):
    from scrub_vision.synth import make_dataset
    make_dataset(tmp_path, n=2, seed=0, val_frac=0.5, poses_source=poses_source)
    assert len(list((tmp_path / "labels").rglob("*.txt"))) == 2
    assert sum(len(p.read_text().splitlines()) for p in (tmp_path / "labels").rglob("*.txt")) >= 3
