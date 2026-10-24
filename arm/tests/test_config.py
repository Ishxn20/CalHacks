from scrub_arm.config import load_arm_config


def test_pick_cfg_defaults(config_dir):
    cfg = load_arm_config(config_dir / "arm.yaml")
    p = cfg.pick
    assert p.mode == "slots"
    assert abs(p.hover - 0.08) < 1e-9
    assert abs(p.descend_step - 0.003) < 1e-9
    assert p.contact_joints == ["joint2", "joint3"]
    assert p.max_declutter_moves == 6
    assert p.perception_url.startswith("http://")
    assert len(p.staging) == 6 and abs(p.staging[0][0] - 0.18) < 1e-9


def test_default_staging_is_reachable_top_down(config_dir):
    import numpy as np
    from scrub_arm.kinematics import Kinematics
    cfg = load_arm_config(config_dir / "arm.yaml")
    k = Kinematics()
    for x, y in cfg.pick.staging:
        seed = np.radians([np.degrees(np.arctan2(y, x)), 60, 70, 0, 50, 0])
        assert k.ik_topdown(x, y, cfg.pick.table_z + cfg.pick.hover, 0.0, seed, (cfg.lower, cfg.upper)) is not None
