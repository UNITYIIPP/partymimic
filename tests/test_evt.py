"""CPU checks; prepare EVT assets with tools/prepare_evt.py before running."""
import copy
import json
import re
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from partymimic.control import control_snapshot, validate_control
from partymimic.dr import XiLayout, ground_geom_ids
from partymimic.evt.motion_io import load_archive
from partymimic.kinematics import Kinematics
from partymimic.mirror import Mirror
from partymimic.motion import MotionLib
from partymimic.robot import get_robot

REPO = Path(__file__).resolve().parents[1]


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        names = [j['name'] for j in json.loads((REPO / 'assets/evt/config.json').read_text())['joints']]
        self.data = dict(root_pos=np.zeros((10, 3), np.float32), rest=np.zeros((10, 68), np.float16),
                         starts=np.array([0, 4]), lengths=np.array([4, 6]), names=np.array(['first', 'second']),
                         joint_names=np.array(names[::-1]), fps=np.array(50.), storage=np.array('compact_v1'),
                         robot=np.array('evt'), quaternion_order=np.array('wxyz'))
        self.data['rest'][:, 0] = 1
        self.data['rest'][:, 4:33] = np.arange(29)[::-1]
        self.data['root_pos'][4:, 0] = 100  # discontinuity between clips must never enter a derivative

    def read(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'clips.npz'
            np.savez(path, **self.data)
            return load_archive(str(path))

    def test_reorder_and_boundaries(self):
        clips = self.read()
        self.assertEqual([n for n, _ in clips], ['first', 'second'])
        self.assertEqual([len(a[0]) for _, a in clips], [4, 6])
        np.testing.assert_array_equal(clips[0][1][2][0], np.arange(29))
        self.assertEqual(clips[1][1][0][0, 0], 100)

    def test_invalid_boundaries_and_quaternions(self):
        for key, value in [('starts', np.array([0, 3])), ('lengths', np.array([4., 6.])),
                           ('quaternion_order', np.array('xyzw'))]:
            with self.subTest(key=key):
                old = self.data[key]
                self.data[key] = value
                with self.assertRaises(ValueError):
                    self.read()
                self.data[key] = old
        self.data['rest'][0, :4] = 0
        with self.assertRaises(ValueError):
            self.read()


@unittest.skipUnless((REPO / 'data/evt/evt.xml').exists(), 'run tools/prepare_evt.py first')
class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.robot = get_robot('evt')
        cls.kin = Kinematics(cls.robot, 'cpu')
        cls.layout = XiLayout(cls.robot, 'cpu')

    def test_fk_matches_mujoco(self):
        model = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(7)
        for _ in range(8):
            data.qpos[:3] = [0.1, -0.2, 0.95]
            data.qpos[3:7] = rng.normal(size=4)
            data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
            data.qpos[7:] = np.array(self.robot.default_joint_pos_list()) + rng.normal(0, .15, 29)
            mujoco.mj_forward(model, data)
            tensor = lambda x: torch.tensor(x, dtype=torch.float32)[None]
            p, R, _, _ = self.kin.forward(tensor(data.qpos[:3]), tensor(data.xmat[1].reshape(3, 3)),
                torch.zeros(1, 3), torch.zeros(1, 3), tensor(data.qpos[7:]), torch.zeros(1, 29))
            ids = [model.body(n).id for n in self.robot.tracked_bodies]
            np.testing.assert_allclose(p[0], data.xpos[ids], atol=2e-6)
            np.testing.assert_allclose(R[0], data.xmat[ids].reshape(-1, 3, 3), atol=2e-6)

    def test_mirror_involution_and_geometry(self):
        mirror = Mirror(self.robot, self.layout.slices, 'cpu')
        torch.manual_seed(2)
        q0 = torch.tensor(self.robot.default_joint_pos_list())
        q = q0 + torch.randn(32, 29) * .2
        torch.testing.assert_close(mirror.joints(mirror.joints(q)), q)
        torch.testing.assert_close(mirror.joints(q0), q0)
        alpha = torch.tensor(self.robot.action_scale_list())
        torch.testing.assert_close(alpha[mirror.perm], alpha)
        S = (torch.zeros(32, 3), torch.eye(3).expand(32, 3, 3), torch.zeros(32, 3), torch.zeros(32, 3), q, torch.zeros_like(q))
        p, _, _, _ = self.kin.forward(*S)
        mp, _, _, _ = self.kin.forward(*mirror.state(S))
        from partymimic.evt.evt_constants import partner
        perm = [self.robot.tracked_bodies.index(partner(n)) for n in self.robot.tracked_bodies]
        # Original CAD yaw offsets mean approximate, not exact, bilateral FK.
        self.assertLess(float((mp - p[:, perm] @ mirror.M).norm(dim=-1).max()), .004)

    def test_motion_boundaries_and_sparse_interface(self):
        z = np.zeros((4, 29), np.float32)
        quat = np.tile([1., 0., 0., 0.], (4, 1))
        clips = [('a', (np.zeros((4, 3)), quat, z, 50.)),
                 ('b', (np.ones((4, 3)) * 100, quat, z, 50.))]
        lib = MotionLib.from_clips(clips, self.kin, 'cpu')
        self.assertEqual(lib.clip_len.tolist(), [4, 4])
        torch.testing.assert_close(lib.root_v, torch.zeros_like(lib.root_v))
        self.assertTrue(bool((lib.future_frames(torch.tensor([2, 3]), 10) <= 3).all()))
        lib.set_interface('sparse5', self.robot.tracked_bodies)
        self.assertEqual(lib.kp, [0, 7, 10, 13, 3, 6])
        self.assertTrue(bool(torch.isfinite(lib.goal(torch.tensor([0]), 10)).all()))

    def test_effective_controls_and_checkpoint_guard(self):
        s = control_snapshot(self.robot)
        params = json.loads((REPO / 'assets/evt/config.json').read_text())['joints']
        for field, source in [('kp', 'stiffness'), ('kd', 'damping'), ('armature', 'armature'), ('alpha', 'action_scale')]:
            np.testing.assert_allclose(s[field], [j[source] for j in params])
        self.assertEqual(self.layout.dim, 33)
        self.assertFalse(self.robot.has_action_delay)
        validate_control({'robot_control': s}, s)
        changed = copy.deepcopy(s)
        changed['armature'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'armature'):
            validate_control({'robot_control': s}, changed)
        with self.assertRaisesRegex(ValueError, 'lacks'):
            validate_control({}, s)

    def test_foot_capsule_lengths_and_support_extent(self):
        model = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
        parts = ('front_outer', 'front_inner', 'mid_outer', 'mid_inner',
                 'strip_outer', 'strip_center', 'strip_inner')
        # Cylinder lengths in tiangong2dex.urdf; capsule length includes end caps.
        lengths = (.07, .07, .24, .24, .25, .26, .25)
        for side in ('left', 'right'):
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for part, length in zip(parts, lengths):
                i = model.geom(f'foot_{side}_{part}_collision').id
                self.assertEqual(model.geom_type[i], mujoco.mjtGeom.mjGEOM_CAPSULE)
                r, h = model.geom_size[i, :2]
                self.assertAlmostEqual(2 * (r + h), length)
                mat = np.empty(9)
                mujoco.mju_quat2Mat(mat, model.geom_quat[i])
                extent = np.abs(mat.reshape(3, 3)[:, 2]) * h + r
                lo = np.minimum(lo, model.geom_pos[i] - extent)
                hi = np.maximum(hi, model.geom_pos[i] + extent)
            np.testing.assert_allclose([lo[0], hi[0]], [-.085, .175], atol=1e-6)
            self.assertAlmostEqual(hi[1] - lo[1], .084, places=6)
            self.assertLess(abs(lo[2] + .058), 1e-6)  # source axes are rounded, not exactly horizontal

    def test_ground_contact_has_friction_with_g1_collision_rules(self):
        from mjlab.scene import Scene
        from partymimic.env import make_scene_cfg
        model = Scene(make_scene_cfg(self.robot, 1), device='cpu').compile()
        ground = ground_geom_ids(model)
        terrain = model.geom('terrain').id
        self.assertIn(terrain, ground)
        self.assertEqual(model.geom_condim[terrain], 3)
        self.assertEqual(model.geom_priority[terrain], 0)
        colliders = [i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/')
                     and re.fullmatch(self.robot.dr.friction_geoms, model.geom(i).name.removeprefix('robot/'))]
        self.assertEqual(len(colliders), 39)
        model.geom_friction[colliders, 0] = .45
        model.geom_friction[ground, 0] = .3
        for name in ('foot_left_strip_center_collision', 'left_tcp_link_collision_0_collision',
                     'knee_pitch_l_link_collision_0_collision', 'pelvis_collision_0_collision'):
            gid = model.geom('robot/' + name).id
            foot = name.startswith('foot_')
            self.assertEqual(model.geom_condim[gid], 3 if foot else 1)
            self.assertEqual(model.geom_priority[gid], 1 if foot else 0)
            data = mujoco.MjData(model)
            data.qpos[7:] = self.robot.default_joint_pos_list()
            mujoco.mj_forward(model, data)
            data.qpos[2] -= data.geom_xpos[gid, 2]  # place chosen geometry across the ground
            mujoco.mj_forward(model, data)
            contacts = [c for c in data.contact if set(c.geom) == {gid, terrain}]
            self.assertTrue(contacts, name)
            for contact in contacts:
                self.assertEqual(contact.dim, 3, name)
                np.testing.assert_allclose(contact.friction[:2], [.45, .45], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
