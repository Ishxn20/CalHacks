"""The B601-RS arm in MuJoCo, built from arm/urdf/ReBot_Arm_RS.urdf.

    python -m sim.arm_model <reBotArm_control_py>/urdf/RS/meshes/shared    (re-vendor meshes + rebuild b601.xml)

Meshes: Seeed's `shared/` STLs (github.com/Seeed-Projects/reBotArm_control_py), decimated to keep the repo
small; the URDF's visual-only parts are dropped. Added on top of the URDF: a table plane at pick.table_z,
a `tool` site at the gripper_end frame (= scrub_arm Kinematics' tool frame), position actuators on
joint1-6 with arm.yaml's MIT kp/kd and limits, and one actuator driving both fingers (coupled).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arm"))

from scrub_arm.config import load_arm_config  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets" / "b601"
XML = ASSETS / "b601.xml"
URDF = ROOT / "arm" / "urdf" / "ReBot_Arm_RS.urdf"
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FACES = 2500


def vendor(src: Path) -> None:
    import trimesh

    (ASSETS / "shared").mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(src).glob("*.STL")):
        m = trimesh.load(f)
        if len(m.faces) > FACES:
            m = m.simplify_quadric_decimation(face_count=FACES)
        m.export(ASSETS / "shared" / f"{f.stem}.stl")


def build() -> None:
    cfg = load_arm_config()
    urdf = URDF.read_text()
    urdf = re.sub(r"<visual>(?:(?!</visual>).)*?meshes/visual/.*?</visual>", "", urdf, flags=re.S)
    urdf = re.sub(r'filename="\.\./meshes/shared/(\w+)\.STL"', r'filename="\1.stl"', urdf)
    urdf = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="%s" discardvisual="false" fusestatic="false"'
                  r' balanceinertia="true"/></mujoco>' % (ASSETS / "shared"), urdf, count=1)
    spec = mujoco.MjSpec.from_string(urdf)
    w = spec.worldbody
    w.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01],
               pos=[0, 0, cfg.pick.table_z], rgba=[0.85, 0.85, 0.82, 1])
    spec.body("gripper_end").add_site(name="tool", size=[0.004] * 3, rgba=[1, 0, 0, 1])
    for b in spec.bodies:
        b.gravcomp = 1.0                 # the real servos hold position; sag isn't what we study here
    for j in cfg.joints:
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=j.kp, kv=j.kd)
        a.ctrlrange = [j.lower, j.upper]
        a.ctrllimited = True
        spec.joint(j.name).armature = 0.05
        spec.joint(j.name).damping = [1.0, 0, 0]
    g = spec.add_actuator(name="gripper", target="gripper_joint1", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    g.set_to_position(kp=200.0, kv=5.0)
    g.ctrlrange, g.ctrllimited = [0, 0.05], True
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="gripper_joint2", name2="gripper_joint1")
    XML.write_text(spec.to_xml().replace(f'meshdir="{ASSETS / "shared"}"', 'meshdir="shared"'))


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(XML))


def arm_qpos_adr(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS])


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, q) -> None:
    """Place joint1-6 at q (rad) and update kinematics (no dynamics)."""
    data.qpos[arm_qpos_adr(model)] = q
    mujoco.mj_kinematics(model, data)


def tool_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    s = data.site("tool")
    return s.xpos.copy(), s.xmat.reshape(3, 3).copy()


def touches_table(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    mujoco.mj_collision(model, data)
    t = model.geom("table").id
    return any(t in (c.geom1, c.geom2) for c in data.contact[:data.ncon])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        vendor(Path(sys.argv[1]))
    build()
    print(f"wrote {XML}")
