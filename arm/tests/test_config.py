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
    assert len(p.staging) == 6 and abs(p.staging[0][0] - 0.25) < 1e-9
