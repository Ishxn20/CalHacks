"""Wire config -> bus -> driver -> assistant together."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bus import RobstrideBus, SimBus
from .catalog import Catalog
from .config import DEFAULT_CONFIG_DIR, ArmConfig, PoseLibrary, TrayConfig, load_arm_config, load_poses, load_tray_config
from .driver import ArmDriver
from .handoff import ScrubAssistant
from .verify import Verifier


@dataclass
class App:
    cfg: ArmConfig
    tray: TrayConfig
    poses: PoseLibrary
    driver: ArmDriver
    catalog: Catalog
    assistant: ScrubAssistant


def build(sim: bool = False, config_dir: Path | str | None = None) -> App:
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    cfg = load_arm_config(d / "arm.yaml")
    tray = load_tray_config(d / "tray.yaml" if config_dir else None)  # default: shared repo-root tray.yaml
    poses = load_poses(d / "poses.yaml")
    if sim:
        start = poses.poses["home"].q if "home" in poses.poses else np.zeros(cfg.n)
        bus = SimBus(cfg, start=np.append(start, cfg.gripper.open))
    else:
        bus = RobstrideBus(cfg)
    driver = ArmDriver(cfg, bus, poses, sim=sim)
    catalog = Catalog(tray, default_orientation=cfg.handoff.default_orientation)
    assistant = ScrubAssistant(driver, catalog, Verifier(cfg.verify))
    return App(cfg, tray, poses, driver, catalog, assistant)


def check_poses(app: App) -> list[str]:
    """Problems that would stop a handoff: missing/untaught poses, out-of-limit poses."""
    needed = {"home", "handoff"} | {p for s in app.tray.slots for p in s.approach}
    problems = []
    for name in sorted(needed):
        if name not in app.poses.poses:
            problems.append(f"missing pose {name}")
        elif not app.poses.poses[name].taught:
            problems.append(f"placeholder pose {name}")
    for name, pose in app.poses.poses.items():
        try:
            app.driver.check_limits(pose.q)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name}: {e}")
    if not app.cfg.gripper.calibrated:
        problems.append("gripper open/closed angles not calibrated")
    return problems
