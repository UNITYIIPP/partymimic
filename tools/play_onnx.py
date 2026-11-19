"""Run an exported policy (tools/export_onnx.py) in the simulator the way a robot controller would.

The basic observation is assembled from raw robot state and the ONNX outputs. Policies with keypoint
errors and robots with a sensing chain use the simulator's observation, which provides the matching
kinematics and estimated signals. The motion plays once from its first frame with no randomization or
noise, as in partymimic.evaluate. Use --record to save the states for tools/render.py.

Usage: python tools/play_onnx.py logs/<run>/dance1_subject2.onnx --robot g1 \
           --motion data/lafan1_g1/dance1_subject2.csv
"""

import argparse
import json

import numpy as np
import onnxruntime as ort
import torch

from partymimic.dr import XiLayout
from partymimic.env import TrackingSim, make_sim
from partymimic.kinematics import Kinematics
from partymimic.mdp import MAX_STEP_REWARD, MDP
from partymimic.motion import CONTROL_DT, MotionLib
from partymimic.robot import get_robot
from partymimic.control import control_snapshot, validate_control
from partymimic.rotations import mat_to_quat


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--robot", type=str, default=None)
    ap.add_argument("--clip", type=str, help="clip name inside a packed --motion archive")
    ap.add_argument("--motion", type=str, required=True, help="the motion baked into the ONNX")
    ap.add_argument("--record", help="save the ONNX rollout for tools/render.py")
    ap.add_argument("--seconds", type=float, help="limit the simulated duration for a preview")
    a = ap.parse_args()
    if a.seconds is not None and (not np.isfinite(a.seconds) or a.seconds <= 0):
        ap.error("--seconds must be positive and finite")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = torch.get_num_threads()
    sess = ort.InferenceSession(a.onnx, sess_options=session_options, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    vec = lambda k: np.array(meta[k].split(","), dtype=np.float32)  # noqa: E731
    q0, H = vec("default_joint_pos"), int(meta["observation_history"])

    dev = "cuda:0"
    robot = get_robot(a.robot or meta.get("robot", "g1"))
    validate_control({"robot_control": json.loads(meta.get("robot_control", "null"))}, control_snapshot(robot), inference=True)
    assert ",".join(robot.joint_names) == meta["joint_names"]
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_single(a.motion, robot, kin, dev, a.clip or meta.get("motion_clip") or None)
    if motion.names[0] != meta["motion"] or motion.num_frames != int(meta["motion_num_frames"]):
        ap.error("--motion and --clip must select the reference used to export this ONNX")
    obs_mode = meta.get("obs_err", "none")
    motion.set_interface("joint", robot.tracked_bodies, obs_mode)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    env = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(1, device=dev), motion.clip_start.clone())

    ref_quat = vec("anchor_quat_w_init")
    a_prev = np.zeros(robot.num_joints, dtype=np.float32)
    hist, ret, obs_err = None, 0.0, 0.0
    use_sim_obs = robot.sensing.stateful or obs_mode == "kp"
    total_steps = int(motion.clip_len[0].item()) - 1
    step_limit = total_steps if a.seconds is None else min(total_steps, int(np.ceil(a.seconds / CONTROL_DT)))
    trajectory = {name: [] for name in ("root_p", "root_quat", "q", "f", "alive")}

    def record_state():
        root_p, root_R, _, _, joint_pos, _ = env.state()
        trajectory["root_p"].append(root_p.cpu())
        trajectory["root_quat"].append(mat_to_quat(root_R).cpu())
        trajectory["q"].append(joint_pos.cpu())
        trajectory["f"].append(env.f.cpu())
        trajectory["alive"].append((~env.done_eval).cpu())

    steps = 0
    while steps < step_limit and not bool(env.done_eval.item()):
        if a.record:
            record_state()
        sim_o = env.observe()[0].cpu().numpy()
        if use_sim_obs:
            o = sim_o
        else:
            # sensors: joint encoders, root attitude and gyro (root frame)
            _, R, _, w, q, qd = (x[0].cpu().numpy() for x in env.state())
            R_rel = R.T @ quat_to_mat(ref_quat)
            o = np.concatenate((q - q0, qd, R.T @ w, -R[2], a_prev, R_rel[:, :2].reshape(6))).astype(np.float32)
            obs_err = max(obs_err, float(np.abs(o - sim_o).max()))
        hist = np.repeat(o[None], H, 0) if hist is None else np.concatenate((hist[1:], o[None]))
        outputs = sess.run(["actions", "next_anchor_quat_w"], {
            "obs_history": hist[None], "time_step": np.array([[env.f.item()]], dtype=np.float32),
        })
        actions, next_ref_quat = outputs
        ret += env.step(torch.from_numpy(actions).to(dev))["r"].item()
        steps += 1
        if a.record and bool(env.failed_eval.item()):
            record_state()
        env.f = torch.minimum(env.f, motion.clip_end - 1)
        a_prev, ref_quat = actions[0], next_ref_quat[0]
    if a.record:
        np.savez_compressed(a.record, names=np.array(motion.names),
                            **{name: torch.stack(values).numpy() for name, values in trajectory.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(),
                            ref_q=motion.q.cpu().numpy(), clip_start=motion.clip_start.cpu().numpy(),
                            clip_len=motion.clip_len.cpu().numpy())
    norm = ret / (step_limit * MAX_STEP_REWARD)
    print(f"{meta['motion']}: {steps}/{step_limit} simulated frames, norm return {norm:.4f}, "
          f"failed {bool(env.failed_eval.item())}"
          + ("" if use_sim_obs else f", max |obs - simulator obs| {obs_err:.2e}"))


if __name__ == "__main__":
    main()
