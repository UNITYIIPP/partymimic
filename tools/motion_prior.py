"""Per-clip motion features for the adaptive clip allocation (train.py --motion_prior, spot.Cover).

    PYTHONPATH=src python tools/motion_prior.py <motion dir> <out.npz> [--robot g1mj] [--glob '*.npz']

Forward kinematics of every clip's reference (MuJoCo, the robot's MJCF; ground at z = 0), per clip:
  unsup     share of frames with no support body near the ground while the centre of mass is not in free fall
            (|a_z + g| > g/2): the motion is held up by an object that is not simulated (a chair, a wall, a partner)
  zmp_out   share of frames in ground contact whose zero-moment point lies > 0.15 m from the contacts (and the
            segments between them): leaning or sitting on something
  air       share of frames with no support body near the ground
  angvel    95th percentile of the root angular speed (rad/s)
  T         frames
Cover reads unsup (a clip is feasible below 0.1) and angvel. Clips are keyed by file name; the "__cN" copies of a clip
share its features, and clips without features count as feasible with the median angular speed. A library retargeted
to several robots can use one robot's file (the homepage RP1 runs use the G1 library's).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from partymimic.motion import load_npz  # noqa: E402
from partymimic.robot import get_robot  # noqa: E402

G = 9.81
# bodies that can bear weight and the height of their frame origin below which they count as on the ground
SUPPORT = {
    "g1": {"left_ankle_roll_link": 0.12, "right_ankle_roll_link": 0.12, "left_wrist_yaw_link": 0.10,
           "right_wrist_yaw_link": 0.10, "left_knee_link": 0.12, "right_knee_link": 0.12, "pelvis": 0.20,
           "torso_link": 0.20},
    "rp1": {"left_ankle_roll_link": 0.12, "right_ankle_roll_link": 0.12, "left_wrist_pitch_link": 0.10,
            "right_wrist_pitch_link": 0.10, "left_knee_link": 0.12, "right_knee_link": 0.12, "base_link": 0.20,
            "waist_yaw_link": 0.20},
}  # fmt: skip
SUPPORT["g1mj"] = SUPPORT["g1"]
SUPPORT["evt"] = {"ankle_roll_l_link": 0.12, "ankle_roll_r_link": 0.12, "wrist_roll_l_link": 0.10,
                  "wrist_roll_r_link": 0.10, "knee_pitch_l_link": 0.12, "knee_pitch_r_link": 0.12,
                  "pelvis": 0.20, "waist_pitch_link": 0.20}
SUPPORT["evt_nominal"] = SUPPORT["evt"]


def seg_dist(p, a, b):
    """Distance from p to the segment a-b (a == b: to the point); the support region of two contacts ~ their segment."""
    u = b - a
    s = np.clip(np.dot(p - a, u) / max(np.dot(u, u), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + s * u)))


def smooth(x, k=5):
    w = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(x, w, mode="same")
    return np.stack([np.convolve(x[:, i], w, mode="same") for i in range(x.shape[1])], 1)


def clip_feats(m, d, ids, thr, pos, quat, q, fps):
    import mujoco

    T = len(q)
    com, bodies = np.zeros((T, 3)), np.zeros((T, len(ids), 3))
    for t in range(T):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[t], quat[t], q[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        com[t] = d.subtree_com[1]
        bodies[t] = d.xpos[ids]
    dt = 1.0 / fps
    a = smooth(np.gradient(np.gradient(smooth(com), dt, axis=0), dt, axis=0))
    contact = bodies[:, :, 2] < thr
    air = ~contact.any(1)
    unsup = air & (np.abs(a[:, 2] + G) > G / 2)
    zmp_out = np.zeros(T, bool)
    for t in np.nonzero(contact.any(1) & (a[:, 2] + G > 2.0))[0]:
        zmp = com[t, :2] - com[t, 2] * a[t, :2] / (a[t, 2] + G)
        pts = bodies[t, contact[t], :2]
        zmp_out[t] = min(seg_dist(zmp, pts[i], pts[j]) for i in range(len(pts)) for j in range(i, len(pts))) > 0.15
    qd = np.sum(quat[1:] * quat[:-1], 1).clip(-1, 1)  # root angular speed from consecutive wxyz quaternions
    w = 2 * np.arccos(np.abs(qd)) / dt
    edge = slice(5, T - 5) if T > 20 else slice(None)  # the smoothing is unreliable at the ends
    return [unsup[edge].mean(), zmp_out[edge].mean(), air[edge].mean(), float(np.percentile(w, 95)) if len(w) else 0.0, T]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("motion_dir", help="directory of clips or an EVT compact_v1 archive")
    ap.add_argument("out")
    ap.add_argument("--robot", default="g1mj")
    ap.add_argument("--glob", default="*.npz", help="motion files; *.npz as mjlab's, anything else the robot's loader")
    a = ap.parse_args()
    import mujoco

    spec = get_robot(a.robot)
    m = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    d = mujoco.MjData(m)
    sup = SUPPORT[a.robot]
    ids, thr = [m.body(b).id for b in sup], np.array(list(sup.values()))
    source = Path(a.motion_dir)
    if source.is_file() and spec.load_motion_archive is not None:
        clips = spec.load_motion_archive(str(source))
        count = len(clips)
    else:
        loader = load_npz if a.glob.endswith(".npz") else spec.load_motion
        files = sorted(source.glob(a.glob))
        if not files:
            raise SystemExit(f"no {a.glob} in {source}")
        count = len(files)
        clips = ((f.stem, loader(str(f))) for f in files)
    names, rows = [], []
    for i, (name, (pos, quat, q, fps)) in enumerate(clips):
        names.append(name)
        rows.append(clip_feats(m, d, ids, thr, np.asarray(pos), np.asarray(quat), np.asarray(q), fps))
        if i % 1000 == 0:
            print(f"{i}/{count}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, names=np.array(names), x=np.array(rows), cols=np.array(["unsup", "zmp_out", "air", "angvel", "T"]))
    x = np.array(rows)
    print(f"{len(names)} clips -> {a.out}: feasible (unsup < 0.1) {np.mean(x[:, 0] < 0.1):.3f}, "
          f"median angvel {np.median(x[:, 3]):.2f} rad/s")


if __name__ == "__main__":
    main()
