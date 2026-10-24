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


def test_default_staging_is_reachable_top_down(config_dir):
    import numpy as np
    from scrub_arm.kinematics import Kinematics
    cfg = load_arm_config(config_dir / "arm.yaml")
    k = Kinematics()
    for x, y in cfg.pick.staging:
        seed = np.radians([np.degrees(np.arctan2(y, x)), 60, 70, 0, 50, 0])
        assert k.ik_topdown(x, y, cfg.pick.table_z + cfg.pick.hover, 0.0, seed, (cfg.lower, cfg.upper)) is not None


def test_staged_instruments_do_not_overlap(config_dir):
    import math
    cfg = load_arm_config(config_dir / "arm.yaml")
    axis = cfg.pick.staging_yaw - math.pi / 2          # instrument axis (yaw is the jaw-opening direction)
    L, W = 0.250, 0.080                                # longest instrument, widest instrument
    def box(x, y):
        ux, uy = math.cos(axis), math.sin(axis)
        return [(x + sx * ux * L / 2 - sy * uy * W / 2, y + sx * uy * L / 2 + sy * ux * W / 2)
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    from itertools import combinations
    def sep(a, b):
        for poly in (a, b):
            for i in range(4):
                (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
                nx, ny = y2 - y1, x1 - x2
                pa = [nx * x + ny * y for x, y in a]
                pb = [nx * x + ny * y for x, y in b]
                if max(pa) <= min(pb) or max(pb) <= min(pa):
                    return True
        return False
    for (x1, y1), (x2, y2) in combinations(cfg.pick.staging, 2):
        assert sep(box(x1, y1), box(x2, y2)), ((x1, y1), (x2, y2))
