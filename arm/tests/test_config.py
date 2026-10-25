from scrub_arm.config import load_arm_config


def test_pick_cfg_defaults(config_dir):
    cfg = load_arm_config(config_dir / "arm.yaml")
    p = cfg.pick
    assert p.mode == "slots"
    assert abs(p.hover - 0.10) < 1e-9     # 10 cm: the visual centring starts there
    assert abs(p.descend_step - 0.003) < 1e-9
    assert p.contact_joints == ["joint2", "joint3"]
    assert p.perception_url.startswith("http://")


def test_gripper_width_table_and_tool_jaws(config_dir):
    import math

    from scrub_arm.config import load_tray_config
    g = load_arm_config(config_dir / "arm.yaml").gripper
    if g.width_table:          # measured on the real gripper: about 0.33 mm per degree, wider = more open
        assert g.angle_for(10) < g.angle_for(30) < g.angle_for(60)
        assert abs(math.degrees(g.angle_for(30) - g.angle_for(10)) - 20 / 0.33) < 8
    tray = load_tray_config(None)
    for iid in ("scalpel_10", "needle_holder_ryder", "forceps_fine_vascular", "scissors_micro_back"):
        o, c = tray.instruments[iid].jaw_mm
        assert 0 <= c < o <= 82
