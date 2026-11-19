"""Nominal EVT controls, shared geometry and isolation from legacy checkpoints."""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from partymimic.control import control_snapshot, validate_control
from partymimic.robot import get_robot
from tools.prepare_evt import prepare

REPO = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def test_nominal_cannot_overwrite_legacy_assets(self):
        for output in (REPO / 'data/evt', REPO / 'data/evt/nested'):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'separate output'):
                prepare(Path('unused'), output, robot='evt_nominal')


@unittest.skipUnless(all((REPO / 'data' / name / 'evt.xml').is_file()
                        for name in ('evt', 'evt_nominal')), 'prepare both EVT configurations first')
class NominalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = get_robot('evt')
        cls.before = control_snapshot(cls.legacy)
        cls.robot = get_robot('evt_nominal')
        cls.snapshot = control_snapshot(cls.robot)

    def test_compiled_controls_and_limits(self):
        params = json.loads((REPO / 'assets/evt_nominal/config.json').read_text())['joints']
        s = self.snapshot
        for field, key in [('kp', 'stiffness'), ('kd', 'damping'), ('armature', 'armature'),
                           ('passive_damping', 'viscous_friction'), ('frictionloss', 'frictionloss')]:
            np.testing.assert_allclose(s[field], [j[key] for j in params])
        for field in ('actuator_force_range', 'joint_force_range'):
            np.testing.assert_allclose(s[field], [[-j['effort_limit'], j['effort_limit']] for j in params])
        limits = dict(zip(self.robot.joint_names, np.asarray(s['actuator_force_range'])[:, 1]))
        for name, limit in [('hip_pitch_l_joint', 200), ('knee_pitch_l_joint', 330),
                            ('ankle_pitch_l_joint', 55), ('ankle_roll_l_joint', 50),
                            ('elbow_yaw_l_joint', 25), ('wrist_roll_r_joint', 25)]:
            self.assertEqual(limits[name], limit)
        np.testing.assert_allclose(s['alpha'], .25)
        np.testing.assert_allclose(s['passive_damping'], 1)
        np.testing.assert_allclose(s['frictionloss'], .1)
        self.assertFalse(self.robot.has_action_delay)

    def test_geometry_and_motion_interface_match_legacy(self):
        a = mujoco.MjModel.from_xml_path(self.legacy.mjcf_path)
        b = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
        # Nominal retains the original torso and pelvis collision shapes.
        shared_shape = np.ones(a.ngeom, dtype=bool)
        for name in ('waist_pitch_link_collision_0_collision', 'pelvis_collision_0_collision'):
            self.assertEqual(a.geom(name).id, b.geom(name).id)
            self.assertEqual(b.geom_type[b.geom(name).id], mujoco.mjtGeom.mjGEOM_BOX)
            shared_shape[a.geom(name).id] = False
        for field in ('geom_type', 'geom_pos', 'geom_quat', 'geom_size'):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(a, field)[shared_shape], getattr(b, field)[shared_shape])
        for field in ('body_mass', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_inertia',
                      'jnt_pos', 'jnt_axis', 'jnt_range', 'geom_bodyid', 'geom_condim', 'geom_priority', 'geom_friction',
                      'geom_contype', 'geom_conaffinity', 'geom_solref', 'geom_solimp'):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        self.assertEqual(self.robot.joint_names, self.legacy.joint_names)
        self.assertEqual(self.robot.default_joint_pos_list(), self.legacy.default_joint_pos_list())
        self.assertEqual(self.robot.tracked_bodies, self.legacy.tracked_bodies)
        self.assertEqual(self.robot.mirror_sign, self.legacy.mirror_sign)
        self.assertEqual(self.robot.dr, self.legacy.dr)
        self.assertIs(self.robot.load_motion_archive, self.legacy.load_motion_archive)

    def test_legacy_unchanged_and_checkpoints_cannot_mix(self):
        self.assertEqual(self.before, control_snapshot(self.legacy))
        validate_control({'robot_control': self.before}, control_snapshot(self.legacy))
        validate_control({'robot_control': self.snapshot}, control_snapshot(self.robot))
        for saved, current in ((self.before, self.snapshot), (self.snapshot, self.before)):
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate_control({'robot_control': saved}, current)


if __name__ == '__main__':
    unittest.main()
