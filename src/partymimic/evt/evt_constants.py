"""EVT from allmimic armature e17f339; PartyMimic simulation and learning recipe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from partymimic.dr import DomainRandCfg
from partymimic.robot import RobotSpec
from .motion_io import load_clip, load_archive

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / 'data/evt'
CONFIG_PATH = REPO / 'assets/evt/config.json'
MJCF_PATH = ASSETS / 'evt.xml'
PARAMS = json.loads(CONFIG_PATH.read_text())
JOINTS = PARAMS['joints']
JOINT_NAMES = tuple(j['name'] for j in JOINTS)
TRACKED_BODIES = (
    'pelvis', 'hip_roll_l_link', 'knee_pitch_l_link', 'ankle_roll_l_link',
    'hip_roll_r_link', 'knee_pitch_r_link', 'ankle_roll_r_link', 'waist_pitch_link',
    'shoulder_roll_l_link', 'elbow_pitch_l_link', 'wrist_roll_l_link',
    'shoulder_roll_r_link', 'elbow_pitch_r_link', 'wrist_roll_r_link',
)


def _validate_assets():
    if not MJCF_PATH.is_file():
        raise FileNotFoundError('EVT assets missing: run python tools/prepare_evt.py --source <allmimic>/assets/robots/evt')
    provenance = json.loads((ASSETS / 'provenance.json').read_text())
    if provenance.get('collision_recipe') != 'urdf_foot_lengths_g1_full_collision_v1':
        raise ValueError('EVT collision assets are outdated: rerun tools/prepare_evt.py')
    if provenance['config_sha256'] != hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest():
        raise ValueError('EVT config changed: rerun tools/prepare_evt.py to update the MJCF')
    if any(any(v != 0 for v in bounds) for bounds in PARAMS['control_delay_steps'].values()):
        raise ValueError('EVT nonzero control delay needs world-model history support')


def get_spec():
    _validate_assets()
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    spec.assets = {p.name: p.read_bytes() for p in (ASSETS / 'meshes').glob('*.STL')}
    return spec


def get_robot_cfg():
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(PARAMS['root_pos']),
            joint_pos={j['name']: j['default_pos'] for j in JOINTS}, joint_vel={'.*': 0.0}),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(BuiltinPositionActuatorCfg(target_names_expr=(j['name'],),
                stiffness=j['stiffness'], damping=j['damping'], effort_limit=j['effort_limit'],
                armature=j['armature'], frictionloss=j['frictionloss']) for j in JOINTS),
            soft_joint_pos_limit_factor=PARAMS['soft_joint_pos_limit_factor']),
    )


def partner(name: str) -> str:
    return name.replace('_l_', '_SIDE_').replace('_r_', '_l_').replace('_SIDE_', '_r_')


def mirror_signs():
    _validate_assets()
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    M = np.diag([1., -1., 1.])
    signs = []
    for name in JOINT_NAMES:
        a, b = M @ data.xaxis[model.joint(name).id], data.xaxis[model.joint(partner(name)).id]
        # The source CAD has a ~0.00134 rad yaw offset, so bilateral axes are
        # approximate mirror images (maximum residual 0.002683), not exact.
        if np.linalg.norm(b + a) < 0.003:
            signs.append(1.)
        elif np.linalg.norm(b - a) < 0.003:
            signs.append(-1.)
        else:
            raise ValueError(f'EVT joint axes are not mirror partners: {name}')
    return tuple(signs)


SPEC = RobotSpec(
    name='evt', mjcf_path=str(MJCF_PATH), joint_names=JOINT_NAMES, tracked_bodies=TRACKED_BODIES,
    default_joint_pos={j['name']: j['default_pos'] for j in JOINTS},
    action_scale={j['name']: j['action_scale'] for j in JOINTS},
    com_body='waist_pitch_link', com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg, motion_dir=str(REPO / 'data/evt_motions'), motion_glob='*.npz',
    load_motion=load_clip, load_motion_archive=load_archive,
    # As for G1, one friction sample per world applies to all contact surfaces.
    dr=DomainRandCfg(friction_geoms=r'.*_collision'),
    mirror_sign=mirror_signs(), mirror_permutation=tuple(JOINT_NAMES.index(partner(n)) for n in JOINT_NAMES),
    feet=('ankle_roll_l_link', 'ankle_roll_r_link'), contact_height=0.07, flat_foot_height=0.0586,
    sparse_bodies=('waist_pitch_link', 'wrist_roll_l_link', 'wrist_roll_r_link',
                   'ankle_roll_l_link', 'ankle_roll_r_link'),
    control_files=tuple(map(str, (CONFIG_PATH, MJCF_PATH, *sorted((ASSETS / 'meshes').glob('*.STL'))))),
)
