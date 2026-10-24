"""Load arm.yaml / tray.yaml / poses.yaml into typed objects.

All angles are stored in degrees on disk (easier to read and hand-edit) and
converted to radians here; everything downstream works in radians.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
# tray.yaml is shared with voice/ (and CV later), so it lives at the repo root.
DEFAULT_TRAY_PATH = Path(__file__).resolve().parents[2] / "config" / "tray.yaml"


@dataclass
class JointCfg:
    name: str
    motor_id: int
    feedback_id: int
    model: str
    lower: float          # rad
    upper: float          # rad
    max_vel: float        # rad/s
    kp: float
    kd: float
    pv_vel_kp: float
    pv_vel_ki: float
    pv_pos_kp: float
    pv_vlim: float


@dataclass
class GripperCfg:
    motor_id: int
    feedback_id: int
    model: str
    open: float           # rad
    closed: float         # rad
    calibrated: bool
    move_kp: float
    move_kd: float
    grip_kp: float
    grip_kd: float
    settle_s: float
    holding_min_gap: float  # rad


@dataclass
class MotionCfg:
    speed_scale: float
    approach_speed_scale: float
    min_move_s: float
    settle_tol: float               # rad
    settle_timeout_s: float
    tracking_error_limit: float | None  # rad


@dataclass
class TugCfg:
    enabled: bool
    joints: list[str]
    deviation: float      # rad
    hold_s: float


@dataclass
class HandoffCfg:
    orientation_offsets: dict[str, float]  # rad
    default_orientation: str
    release_wait_s: float
    present_timeout_s: float
    tug: TugCfg


@dataclass
class VerifyCfg:
    url: str | None
    timeout_s: float
    min_confidence: float
    fail_open: bool


@dataclass
class PickCfg:
    mode: str                      # "slots" | "ik"
    reach: str                     # "ik" | "rl": how ik mode turns a tool pose into joints
    perception_url: str
    table_z: float                 # m
    hover: float                   # m
    descend_step: float            # m
    contact_deviation: float       # rad
    contact_joints: list[str]
    max_declutter_moves: int
    staging: list[tuple[float, float]]   # m
    staging_yaw: float             # rad


@dataclass
class ArmConfig:
    channel: str
    control_rate_hz: float
    arm_mode: str
    joints: list[JointCfg]
    gripper: GripperCfg
    motion: MotionCfg
    handoff: HandoffCfg
    verify: VerifyCfg
    pick: PickCfg
    server_host: str
    server_port: int
    active_report: bool = True
    path: Path | None = None

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints])

    @property
    def max_vel(self) -> np.ndarray:
        return np.array([j.max_vel for j in self.joints])

    def joint_index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(f"unknown joint {name!r}")


@dataclass
class Instrument:
    id: str
    name: str
    category: str
    orientable: bool
    aliases: list[str]


@dataclass
class Slot:
    id: str
    instrument: str
    suture: str | None
    size: str | None
    approach: list[str]


@dataclass
class TrayConfig:
    instruments: dict[str, Instrument]
    sutures: dict[str, list[str]]
    slots: list[Slot]


@dataclass
class Pose:
    q: np.ndarray         # rad
    taught: bool


@dataclass
class PoseLibrary:
    path: Path
    poses: dict[str, Pose] = field(default_factory=dict)

    def get(self, name: str) -> Pose:
        if name not in self.poses:
            raise KeyError(f"pose {name!r} not found in {self.path.name}; teach it first")
        return self.poses[name]

    def set(self, name: str, q: np.ndarray) -> None:
        self.poses[name] = Pose(q=np.array(q, dtype=float), taught=True)

    def delete(self, name: str) -> None:
        self.poses.pop(name, None)

    def save(self) -> None:
        data = {
            "poses": {
                name: {"joints_deg": [round(math.degrees(x), 2) for x in p.q], "taught": p.taught}
                for name, p in self.poses.items()
            }
        }
        header = (
            "# Taught joint-space poses (degrees, joint1..joint6).\n"
            "# Generated/updated by `python -m scrub_arm teach`. Poses with taught: false are\n"
            "# PLACEHOLDERS so the simulator runs; the real arm refuses to move to them.\n"
        )
        _atomic_write(self.path, header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None))


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _rad(x: float) -> float:
    return math.radians(float(x))


def load_arm_config(path: Path | str | None = None) -> ArmConfig:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "arm.yaml"
    d = yaml.safe_load(path.read_text())

    joints = []
    for j in d["joints"]:
        lo, hi = j["limits_deg"]
        mit, pv = j.get("mit", {}), j.get("pos_vel", {})
        joints.append(JointCfg(
            name=j["name"], motor_id=int(j["motor_id"]), feedback_id=int(j["feedback_id"]),
            model=str(j["model"]), lower=_rad(lo), upper=_rad(hi), max_vel=_rad(j["max_vel_deg_s"]),
            kp=float(mit.get("kp", 0.0)), kd=float(mit.get("kd", 0.0)),
            pv_vel_kp=float(pv.get("vel_kp", 0.0)), pv_vel_ki=float(pv.get("vel_ki", 0.0)),
            pv_pos_kp=float(pv.get("pos_kp", 0.0)), pv_vlim=float(pv.get("vlim", 2.0)),
        ))

    g = d["gripper"]
    gripper = GripperCfg(
        motor_id=int(g["motor_id"]), feedback_id=int(g["feedback_id"]), model=str(g["model"]),
        open=_rad(g["open_deg"]), closed=_rad(g["closed_deg"]), calibrated=bool(g.get("calibrated", False)),
        move_kp=float(g["move_kp"]), move_kd=float(g["move_kd"]),
        grip_kp=float(g["grip_kp"]), grip_kd=float(g["grip_kd"]),
        settle_s=float(g["settle_s"]), holding_min_gap=_rad(g["holding_min_gap_deg"]),
    )

    m = d["motion"]
    tel = m.get("tracking_error_limit_deg")
    motion = MotionCfg(
        speed_scale=float(m["speed_scale"]), approach_speed_scale=float(m["approach_speed_scale"]),
        min_move_s=float(m["min_move_s"]), settle_tol=_rad(m["settle_tol_deg"]),
        settle_timeout_s=float(m["settle_timeout_s"]),
        tracking_error_limit=_rad(tel) if tel is not None else None,
    )

    h = d["handoff"]
    t = h.get("tug", {})
    handoff = HandoffCfg(
        orientation_offsets={k: _rad(v) for k, v in h["orientation_offsets_deg"].items()},
        default_orientation=str(h["default_orientation"]),
        release_wait_s=float(h["release_wait_s"]),
        present_timeout_s=float(h["present_timeout_s"]),
        tug=TugCfg(
            enabled=bool(t.get("enabled", False)), joints=list(t.get("joints", [])),
            deviation=_rad(t.get("deviation_deg", 4.0)), hold_s=float(t.get("hold_s", 0.15)),
        ),
    )

    v = d.get("verify", {}) or {}
    verify = VerifyCfg(
        url=v.get("url"), timeout_s=float(v.get("timeout_s", 2.0)),
        min_confidence=float(v.get("min_confidence", 0.6)), fail_open=bool(v.get("fail_open", True)),
    )

    pk = d.get("pick", {}) or {}
    pick = PickCfg(
        mode=str(pk.get("mode", "slots")).lower(), reach=str(pk.get("reach", "ik")).lower(),
        perception_url=str(pk.get("perception_url", "http://127.0.0.1:8770")),
        table_z=float(pk.get("table_z_mm", 0.0)) / 1000.0, hover=float(pk.get("hover_mm", 80.0)) / 1000.0,
        descend_step=float(pk.get("descend_step_mm", 3.0)) / 1000.0,
        contact_deviation=_rad(pk.get("contact_deviation_deg", 1.5)),
        contact_joints=[str(j) for j in pk.get("contact_joints", ["joint2", "joint3"])],
        max_declutter_moves=int(pk.get("max_declutter_moves", 6)),
        staging=[(float(x) / 1000.0, float(y) / 1000.0) for x, y in pk.get("staging_mm", [])],
        staging_yaw=_rad(pk.get("staging_yaw_deg", 0.0)),
    )

    s = d.get("server", {}) or {}
    cfg = ArmConfig(
        channel=str(d["channel"]), control_rate_hz=float(d["control_rate_hz"]),
        arm_mode=str(d.get("arm_mode", "mit")).lower(), joints=joints, gripper=gripper,
        motion=motion, handoff=handoff, verify=verify, pick=pick,
        server_host=str(s.get("host", "127.0.0.1")), server_port=int(s.get("port", 8765)),
        active_report=bool(d.get("active_report", True)), path=path,
    )
    if cfg.arm_mode not in ("mit", "pos_vel"):
        raise ValueError(f"arm_mode must be 'mit' or 'pos_vel', got {cfg.arm_mode!r}")
    if cfg.pick.mode not in ("slots", "ik"):
        raise ValueError(f"pick.mode must be 'slots' or 'ik', got {cfg.pick.mode!r}")
    if cfg.pick.reach not in ("ik", "rl"):
        raise ValueError(f"pick.reach must be 'ik' or 'rl', got {cfg.pick.reach!r}")
    return cfg


def save_gripper_calibration(cfg: ArmConfig, open_rad: float | None, closed_rad: float | None) -> None:
    """Write measured gripper endpoints back into arm.yaml, keeping comments intact."""
    assert cfg.path is not None
    lines = cfg.path.read_text().splitlines()
    out, in_gripper = [], False
    for line in lines:
        stripped = line.strip()
        if not line.startswith(" ") and stripped:
            in_gripper = stripped.startswith("gripper:")
        if in_gripper and open_rad is not None and stripped.startswith("open_deg:"):
            line = f"  open_deg: {math.degrees(open_rad):.2f}"
        elif in_gripper and closed_rad is not None and stripped.startswith("closed_deg:"):
            line = f"  closed_deg: {math.degrees(closed_rad):.2f}"
        elif in_gripper and stripped.startswith("calibrated:"):
            line = "  calibrated: true"
        out.append(line)
    _atomic_write(cfg.path, "\n".join(out) + "\n")
    if open_rad is not None:
        cfg.gripper.open = open_rad
    if closed_rad is not None:
        cfg.gripper.closed = closed_rad
    cfg.gripper.calibrated = True


def load_tray_config(path: Path | str | None = None) -> TrayConfig:
    path = Path(path) if path else DEFAULT_TRAY_PATH
    d = yaml.safe_load(path.read_text())
    instruments = {
        iid: Instrument(
            id=iid, name=str(v.get("name", iid)), category=str(v.get("category", iid)),
            orientable=bool(v.get("orientable", False)), aliases=[str(a) for a in v.get("aliases", [])],
        )
        for iid, v in d["instruments"].items()
    }
    slots = []
    for s in d["slots"]:
        if s["instrument"] not in instruments:
            raise ValueError(f"slot {s['id']}: unknown instrument {s['instrument']!r}")
        slots.append(Slot(
            id=str(s["id"]), instrument=str(s["instrument"]),
            suture=str(s["suture"]) if s.get("suture") else None,
            size=str(s["size"]) if s.get("size") else None,
            approach=[str(p) for p in s["approach"]],
        ))
    sutures = {k: [str(a) for a in v] for k, v in (d.get("sutures") or {}).items()}
    return TrayConfig(instruments=instruments, sutures=sutures, slots=slots)


def load_poses(path: Path | str | None = None) -> PoseLibrary:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "poses.yaml"
    lib = PoseLibrary(path=path)
    if path.exists():
        d = yaml.safe_load(path.read_text()) or {}
        for name, p in (d.get("poses") or {}).items():
            lib.poses[str(name)] = Pose(
                q=np.radians(np.array(p["joints_deg"], dtype=float)), taught=bool(p.get("taught", False))
            )
    return lib
