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
from .reach import ReachMap, load_reach
from .verify import Verifier
from .vision import VisionClient


@dataclass
class App:
    cfg: ArmConfig
    tray: TrayConfig
    poses: PoseLibrary
    driver: ArmDriver
    catalog: Catalog
    assistant: ScrubAssistant
    vision: VisionClient | None
    reach: ReachMap


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
    reach = load_reach(d / "reach_map.yaml")
    vision = VisionClient(cfg.pick) if cfg.pick.vision_url else None
    assistant = ScrubAssistant(driver, catalog, Verifier(cfg.verify), vision=vision, reach=reach)
    return App(cfg, tray, poses, driver, catalog, assistant, vision, reach)


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


def check_pick(app: App) -> list[str]:
    """What `/pick` still needs. Informational: handoffs from taught slots work without any of it."""
    problems = []
    view = app.cfg.pick.view_pose
    if view not in app.poses.poses or not app.poses.poses[view].taught:
        problems.append(f"pick: pose {view} not taught")
    problems += app.reach.ready()
    if app.cfg.pick.marker_mm is None:
        problems.append("pick: marker_mm not set (width/clearance checks need it)")
    return problems
