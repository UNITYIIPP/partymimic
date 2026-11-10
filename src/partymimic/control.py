"""Portable snapshots of effective robot controls for resume, evaluation and export."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from mjlab.entity import Entity

from .robot import RobotSpec


def control_snapshot(robot: RobotSpec) -> dict | None:
    if not robot.control_files:
        return None  # historical G1/RP1 checkpoints did not record these fields
    from .env import PHYSICS_DT
    from .motion import CONTROL_DT

    cfg = robot.robot_cfg()
    model = Entity(cfg).spec.compile()
    jids = [model.joint(n).id for n in robot.joint_names]
    dofs = model.jnt_dofadr[jids]
    aids = []
    for jid in jids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
        if len(matches) != 1:
            raise ValueError('Control snapshot needs one actuator per joint')
        aids.append(int(matches[0]))
    files = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in robot.control_files}
    return {
        'version': 1, 'robot': robot.name, 'joint_names': list(robot.joint_names),
        'kp': model.actuator_gainprm[aids, 0].tolist(),
        'kd': (-model.actuator_biasprm[aids, 2]).tolist(),
        'actuator_force_range': model.actuator_forcerange[aids].tolist(),
        'joint_force_range': model.jnt_actfrcrange[jids].tolist(),
        'joint_limits': model.jnt_range[jids].tolist(),
        'armature': model.dof_armature[dofs].tolist(),
        'passive_damping': model.dof_damping[dofs].tolist(),
        'frictionloss': model.dof_frictionloss[dofs].tolist(),
        'q0': robot.default_joint_pos_list(), 'alpha': robot.action_scale_list(),
        'root_pos': list(cfg.init_state.pos),
        'soft_limit_factor': cfg.articulation.soft_joint_pos_limit_factor,
        'delay_steps': [[a.delay_min_lag, a.delay_max_lag] for a in cfg.articulation.actuators],
        'physics_dt': PHYSICS_DT, 'control_dt': CONTROL_DT,
        'feet': list(robot.feet), 'contact_height': robot.contact_height,
        'flat_foot_height': robot.flat_foot_height, 'asset_sha256': files,
        'friction_randomization': {'geoms': robot.dr.friction_geoms,
                                   'range': list(robot.dr.friction_range),
                                   'ground_selection': 'world_welded_bodies'},
    }


def validate_control(checkpoint: dict, current: dict | None) -> None:
    saved = checkpoint.get('robot_control')
    if saved != current:
        if saved is None:
            raise ValueError('Checkpoint lacks robot_control; cannot verify EVT dynamics')
        if current is None:
            raise ValueError('Checkpoint robot_control does not match the selected robot')
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        raise ValueError(f'Robot control differs from checkpoint: {", ".join(changed)}')
