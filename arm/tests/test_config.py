from scrub_arm.config import load_arm_config


def test_pick_cfg_defaults(config_dir):
    cfg = load_arm_config(config_dir / "arm.yaml")
    p = cfg.pick
    assert p.mode == "slots"
    assert abs(p.hover - 0.10) < 1e-9     # 10 cm: the visual centring starts there
    assert abs(p.descend_step - 0.003) < 1e-9
    assert p.contact_joints == ["joint2", "joint3"]
    assert p.perception_url.startswith("http://")
