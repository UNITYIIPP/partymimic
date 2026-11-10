"""Left-right (sagittal-plane) mirror of states, actions and sim context, for world-model data augmentation.

The mirror is the reflection M = diag(1, -1, 1) of the world y axis: positions and linear velocities
map to M x, rotations to M R M, angular velocities (pseudo-vectors) to -M w, and joints to their
left/right partner, with the sign flipped for roll and yaw joints (and the waist's roll and yaw).
"""

from __future__ import annotations

import torch

from .robot import RobotSpec

_FLIP_TYPES = ("roll", "yaw")


class Mirror:
    def __init__(self, robot: RobotSpec, xi_slices: dict, device: str):
        names = list(robot.joint_names)
        perm, sign = [], []
        for i, n in enumerate(names):
            partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(robot.mirror_permutation[i] if robot.mirror_permutation is not None else names.index(partner))
            sign.append(-1.0 if any(f"_{t}_" in n for t in _FLIP_TYPES) else 1.0)
        self.perm = torch.tensor(perm, device=device)
        self.sign = torch.tensor(robot.mirror_sign or sign, device=device)
        if sorted(perm) != list(range(len(names))) or any(perm[perm[i]] != i for i in range(len(names))):
            raise ValueError("Joint mirror must be an involutive permutation")
        if not torch.equal(self.sign * self.sign[self.perm], torch.ones_like(self.sign)):
            raise ValueError("Joint mirror signs must cancel after two reflections")
        self.M = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device))
        self.xi_slices = xi_slices
        unknown = set(xi_slices) - {"friction", "joint_offset", "com"}
        assert not unknown, f"mirror of sim-context entries {unknown} not defined"

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.sign

    def state(self, S):
        p, R, v, w, q, qd = S
        M = self.M
        return (p @ M, M @ R @ M, v @ M, -(w @ M), self.joints(q), self.joints(qd))

    def xi(self, xi: torch.Tensor) -> torch.Tensor:
        out = xi.clone()
        sl = self.xi_slices["joint_offset"]
        out[..., sl] = self.joints(xi[..., sl])
        c = self.xi_slices["com"]
        out[..., c.start + 1] = -xi[..., c.start + 1]  # lateral COM offset
        return out
