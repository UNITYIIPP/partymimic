"""XtremeMotion, the curated motion subset of OmniXtreme (https://github.com/Perkins729/OmniXtreme, Apache-2.0), as 14
G1 training clips. The release is one 50 Hz BeyondMimic npz in which the motions are joined by exactly static standing
frames; it is cut at those, each clip keeping 0.5 s of the standing pose on both sides and starting at the origin.

    python xtreme.py <release npz> <out dir>     -> <out>/xtreme_XX__c1.npz (motion npz as prep_motions.py writes)

The "__c1" suffix (one copy of each clip) keeps the clip names of the project page's library.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from prep_motions import G1_XML, convert

# Isaac Lab (BeyondMimic) joint order -> MuJoCo order, from OmniXtreme's deploy_mujoco.py
PERM = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
PAD, MIN_STATIC, MIN_LEN = 0.5, 10, 1.5  # s, frames, s


def segments(d: dict, fps: int) -> list[tuple[int, int]]:
    v = np.linalg.norm(d["body_lin_vel_w"][:, 0], axis=1)
    qd = np.abs(d["joint_vel"]).max(1)
    static = (v < 1e-3) & (qd < 1e-3)
    runs, s = [], None  # [start, end) of static runs
    for i, st in enumerate(np.append(static, False)):
        if st and s is None:
            s = i
        elif not st and s is not None:
            if i - s >= MIN_STATIC:
                runs.append((s, i))
            s = None
    pad = int(PAD * fps)
    bounds = [0] + [x for r in runs for x in r] + [len(static)]
    return [(max(0, a - pad), min(len(static), b + pad))  # moving stretches between static runs
            for a, b in zip(bounds[0::2], bounds[1::2]) if (b - a) / fps >= MIN_LEN]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    d = dict(np.load(a.src))
    fps = int(d["fps"][0])
    qpos = np.concatenate([d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"][:, PERM]], 1).astype(np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    seg = segments(d, fps)
    for i, (s, e) in enumerate(seg):
        q = qpos[s:e].copy()
        q[:, :2] -= q[0, :2]
        np.savez(out / f"xtreme_{i:02d}__c1.npz", **convert(q.astype(np.float64), model, float(fps)))
        print(f"xtreme_{i:02d}: frames {s}-{e} ({(e - s) / fps:.1f} s), root height min {q[:, 2].min():.2f} m")
    print(f"{len(seg)} clips, {sum(e - s for s, e in seg) / fps / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
