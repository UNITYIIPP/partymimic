"""Import local EVT assets and build PartyMimic's robot-only MJCF.

python tools/prepare_evt.py --source /path/to/allmimic/assets/robots/evt
Generated assets live in data/evt (not redistributed in git).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
FOOT_PARTS = ('front_outer', 'front_inner', 'mid_outer', 'mid_inner',
              'strip_outer', 'strip_center', 'strip_inner')


def match_foot_lengths(world: ET.Element, reference_urdf: Path) -> None:
    """Match cylinder end-to-end lengths, including both capsule hemispheres.

    The reference has seven cylinders per foot, in FOOT_PARTS order. Retain the
    training model's capsule radii and centers (hence foot width and sole height).
    """
    urdf = ET.parse(reference_urdf).getroot()
    geoms = {g.get('name'): g for g in world.iter('geom')}
    for side, short in (('left', 'l'), ('right', 'r')):
        link = urdf.find(f"link[@name='ankle_roll_{short}_link']")
        collisions = [] if link is None else link.findall('collision')
        if len(collisions) != len(FOOT_PARTS):
            raise ValueError(f'Expected seven reference foot cylinders: {reference_urdf}, {side}')
        for part, collision in zip(FOOT_PARTS, collisions):
            cylinder = collision.find('geometry/cylinder')
            geom = geoms[f'foot_{side}_{part}_collision']
            if cylinder is None or geom.get('type') != 'capsule':
                raise ValueError(f'Expected cylinder reference and capsule target: {side}/{part}')
            length = float(cylinder.get('length'))
            radius = float(geom.get('size').split()[0])
            half_segment = length / 2 - radius
            if half_segment <= 0:
                raise ValueError(f'Reference cylinder is too short for capsule radius: {side}/{part}')
            geom.set('size', f'{radius:.9g} {half_segment:.9g}')


def prepare(source: Path, output: Path, reference_urdf: Path | None = None) -> None:
    config_path = REPO / 'assets/evt/config.json'
    config = json.loads(config_path.read_text())
    xml = source / 'urdf/evt2_simple.xml'
    root = ET.parse(xml).getroot()
    for tag in ('actuator', 'sensor', 'keyframe'):
        for node in root.findall(tag):
            root.remove(node)
    option = root.find('option')
    if option is not None:
        option.attrib.pop('actuatorgroupdisable', None)
    root.find('compiler').set('meshdir', 'meshes')
    world = root.find('worldbody')
    reference_urdf = reference_urdf or source / 'urdf/tiangong2dex.urdf'
    for geom in list(world.findall('geom')):
        if geom.get('name') == 'ground':
            world.remove(geom)
    joints = {j['name']: j for j in config['joints']}
    for joint in world.iter('joint'):
        if joint.get('type') == 'free':
            continue
        j = joints[joint.get('name')]
        joint.set('damping', str(j['viscous_friction']))
        joint.set('frictionloss', str(j['frictionloss']))
        joint.set('armature', str(j['armature']))
        joint.set('actuatorfrclimited', 'true')
        joint.set('actuatorfrcrange', f"{-j['effort_limit']} {j['effort_limit']}")
    for i, geom in enumerate(world.iter('geom')):
        if int(geom.get('contype', '1')) or int(geom.get('conaffinity', '1')):
            name = geom.get('name', f'geom_{i}')
            if not name.endswith('_collision'):
                name += '_collision'
            geom.set('name', name)
            foot = name.startswith(('foot_left_', 'foot_right_'))
            # Match mjlab G1 FULL_COLLISION: normal-only body self-contact;
            # body-ground contact is 3D because the ground has condim=3, priority=0.
            geom.set('contype', '1')
            geom.set('conaffinity', '1')
            geom.set('condim', '3' if foot else '1')
            geom.set('priority', '1' if foot else '0')
            geom.set('friction', '0.6 0.005 0.0001' if foot else '1 0.005 0.0001')
    match_foot_lengths(world, reference_urdf)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'meshes').mkdir(exist_ok=True)
    for mesh in root.findall('./asset/mesh'):
        name = Path(mesh.get('file')).name
        mesh.set('file', name)
        shutil.copy2(source / 'meshes' / name, output / 'meshes' / name)
    ET.indent(root)
    ET.ElementTree(root).write(output / 'evt.xml', encoding='utf-8', xml_declaration=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = {'source_xml': str(xml.resolve()), 'source_xml_sha256': digest(xml),
                  'config_sha256': digest(config_path),
                  'foot_length_reference': str(reference_urdf.resolve()),
                  'foot_length_reference_sha256': digest(reference_urdf),
                  'collision_recipe': 'urdf_foot_lengths_g1_full_collision_v1'}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(output / 'evt.xml'))
    actual = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if actual != list(joints) or model.nu != 0:
        raise ValueError('EVT model joint order or actuator cleanup failed')
    print(f'{output / "evt.xml"}: {len(actual)} joints, {model.body_mass.sum():.6f} kg; PD added by EntityCfg')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=REPO / 'data/evt')
    ap.add_argument('--reference-urdf', type=Path, help='foot cylinder lengths; default: <source>/urdf/tiangong2dex.urdf')
    a = ap.parse_args()
    prepare(a.source, a.output, a.reference_urdf)
