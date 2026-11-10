"""Read EVT compact_v1 clips without joining unrelated motion boundaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from partymimic.robot import MotionArrays


def load_archive(path: str) -> list[tuple[str, MotionArrays]]:
    config = json.loads((Path(__file__).resolve().parents[3] / 'assets/evt/config.json').read_text())
    canonical = [j['name'] for j in config['joints']]
    with np.load(path, allow_pickle=False) as d:
        required = {'root_pos', 'rest', 'fps', 'joint_names', 'storage', 'robot'}
        if missing := required - set(d.files):
            raise ValueError(f'{path}: missing compact_v1 fields {sorted(missing)}')
        if str(d['storage'].item()) != 'compact_v1' or str(d['robot'].item()) not in ('evt', 'tiangong_evt2'):
            raise ValueError(f'{path}: expected EVT compact_v1')
        for key, expected in (('quaternion_order', 'wxyz'), ('root_body', 'pelvis'),
                              ('velocity_frame', 'world_link_origin')):
            if key in d and str(d[key].item()) != expected:
                raise ValueError(f'{path}: {key} must be {expected}')
        fps = float(d['fps'].item())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'{path}: invalid fps {fps}')
        names = d['joint_names'].tolist()
        if len(names) != 29 or set(names) != set(canonical):
            raise ValueError(f'{path}: joint_names must contain each EVT joint once')
        perm = [names.index(n) for n in canonical]
        p, rest = d['root_pos'].astype(np.float32), d['rest'].astype(np.float32)
        if p.ndim != 2 or p.shape[1] != 3 or rest.shape != (len(p), 68):
            raise ValueError(f'{path}: invalid root_pos/rest shapes')
        if not np.isfinite(p).all() or not np.isfinite(rest).all():
            raise ValueError(f'{path}: nonfinite motion data')
        quat = rest[:, :4].copy()
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        if not np.allclose(norms, 1, atol=2e-3):
            raise ValueError(f'{path}: invalid root quaternions')
        quat /= norms
        q = rest[:, 4:33][:, perm].copy()
        if ('starts' in d) != ('lengths' in d):
            raise ValueError(f'{path}: starts and lengths must occur together')
        if 'starts' in d:
            starts, lengths = d['starts'], d['lengths']
            clip_names = d['names'].tolist() if 'names' in d else [f'{Path(path).stem}_{i}' for i in range(len(starts))]
        else:
            starts, lengths = np.array([0]), np.array([len(p)])
            clip_names = [str(d['name'].item()) if 'name' in d else Path(path).stem]
        if (starts.ndim != 1 or lengths.shape != starts.shape or not len(starts)
                or not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths < 2) or not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])
                or int(lengths.sum()) != len(p) or len(clip_names) != len(starts)
                or len(set(clip_names)) != len(clip_names)):
            raise ValueError(f'{path}: invalid or duplicate clip boundaries/names')
    return [(name, (p[s:s+n], quat[s:s+n], q[s:s+n], fps))
            for name, s, n in zip(clip_names, starts, lengths)]


def load_clip(path: str) -> MotionArrays:
    with np.load(path, allow_pickle=False) as d:
        native = 'body_pos_w' in d
    if native:
        from partymimic.motion import load_npz
        return load_npz(path)
    clips = load_archive(path)
    if len(clips) != 1:
        raise ValueError(f'{path}: contains {len(clips)} clips; use --motion_file for a packed archive')
    return clips[0][1]
