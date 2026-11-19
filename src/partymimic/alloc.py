"""Dataset-adaptive allocation of the training samples (resets and imagination starts).

Groups are clip segments of at most ``seg`` one-second bins (any motion library has clips and bins). Every collection
transition updates the group the robot is in with its loss l = (1 - clamp(r / r_max, 0, 1)) + beta * terminated:

  * fast / slow averages of l indexed by visits: the running mean of the first 1 / alpha visits, an EMA of rate alpha
    afterwards (no memory of the early-training losses of rarely visited groups), so slow - fast is the progress per
    sample of the recent visits;
  * deficit d = fast average above the library's attainable loss (its 5th percentile over groups);
  * the recent allocation a: the reset share the group received, summed over the collection steps with a half-life
    of rs_half_steps (a measure of how hard the group has been tried lately), and a visit-rate EMA v.

Progress is judged over the second half of the training so far: lp = slow(t/2) - slow(t), from snapshots of the slow
average (taken by tick() outside the captured graphs; 32 of them, thinned to every other one and spaced twice as far
when full, so the one nearest t/2 is never more than 1/32 of the run off). Hard groups that can be learned improve
slowly but steadily; over a short window their progress drowns in the noise (and they would be dropped). The noise
of lp is measured on the library (the spread of the groups that get worse) and lp counts only above z x the noise. A
group is stuck when it was tried (its visits in the window at least tried_mult x the library mean) and still shows
no progress. Every other group is ranked by deficit and gets rank^-rank_pow of the adaptive share (prioritized level
replay: the top 1% of the groups get 1%^(1 - rank_pow) of it, whatever the loss scale of the dataset). A stuck group
drops to the floor; once its visits in the sliding window fall below the mean it is ranked (= tried) again: nothing
is dropped for good. Each group is capped at cap_mult x the uniform share (the excess goes by the floor); a group's
share is uniform over its bins. Imagination starts are re-weighted by reset share / visit rate, which removes the
survival bias of the replay buffer (hard groups end their episodes early). Everything but tick() is fixed-size device
math (no host sync), so it runs inside the captured collection and policy-update graphs."""

from __future__ import annotations

import torch


class Allocator:
    def __init__(self, motion, device: str, seg: int = 16, alpha_fast: float = 1 / 256, alpha_slow: float = 1 / 2048,
                 beta: float = 5.0, rs_half_steps: float = 2000.0, tried_mult: float = 1.0, cap_mult: float = 50.0,
                 r_max: float = 1.0, z: float = 2.0, rank_pow: float = 0.75, gate: bool = True,
                 floor_bins: torch.Tensor | None = None, snap_every: int = 125, snap_k: int = 32,
                 min_window: int = 1000):  # fmt: skip
        m = motion
        self.device, self.r_max, self.beta = device, r_max, beta
        self.af, self.as_, self.z, self.rank_pow, self.gate, self.tried_mult = alpha_fast, alpha_slow, z, rank_pow, gate, tried_mult
        self.a_decay = 0.5 ** (1.0 / rs_half_steps)
        # bin -> group: consecutive bins of a clip, at most seg per group
        bc = m.bin_clip.cpu()
        grp, g, run = [], -1, 0
        for b in range(len(bc)):
            if b == 0 or bc[b] != bc[b - 1] or run == seg:
                g, run = g + 1, 0
            grp.append(g)
            run += 1
        self.bin_group = torch.tensor(grp, device=device)
        self.G = g + 1
        self.cap = min(1.0, cap_mult / self.G)  # per-group share cap: cap_mult x the uniform share
        self.group_clip = torch.zeros(self.G, dtype=torch.long, device=device).index_reduce_(
            0, self.bin_group, m.bin_clip.to(device), "amax", include_self=False)
        self.group_bins = torch.zeros(self.G, device=device).index_add_(0, self.bin_group, torch.ones(len(bc), device=device))
        self.frame_group = self.bin_group[m.frame_bin.to(device)]
        fb = floor_bins if floor_bins is not None else torch.ones(len(bc), device=device)
        self.floor = torch.zeros(self.G, device=device).index_add_(0, self.bin_group, fb.to(device).float())
        self.floor /= self.floor.sum()  # the reset floor's share per group (fallback when no group qualifies)
        self.rank_w = torch.arange(1, self.G + 1, device=device, dtype=torch.float32) ** -rank_pow
        z = lambda v=0.0: torch.full((self.G,), v, device=device)  # noqa: E731
        self.fast, self.slow = z(), z()
        self.N = z()  # visits so far
        self.a = z()  # recent allocation (discounted sum of the reset share)
        self.v = self.group_bins / self.group_bins.sum()  # visit-rate EMA (starts at the duration share)
        self.l_mean = torch.full((), 0.5, device=device)
        self.noise = torch.full((), 0.01, device=device)  # last noise level of the window progress
        self.target = self.v.clone()  # last adaptive reset share per group (diagnostics / starts)
        self.mixed = self.v.clone()  # last full reset share per group (after the floor and failure mixing)
        # snapshots of (slow, N) for the half-training window; the reference is the one nearest to half the iteration
        self.snap_k, self.min_window = snap_k, min_window
        self.snap_every = torch.tensor(snap_every)
        self.snap_t = torch.full((snap_k,), -1, dtype=torch.long)  # host: iteration of each slot, -1 empty
        self.ring_slow = torch.zeros(snap_k, self.G, device=device)
        self.ring_N = torch.zeros(snap_k, self.G, device=device)
        self.ref_slow, self.ref_N = z(), z()
        self.gate_on = torch.zeros((), device=device)  # 1 once the window spans min_window iterations

    # ---------------------------------------------------------------- statistics
    @torch.no_grad()
    def observe(self, f: torch.Tensor, r: torch.Tensor, term: torch.Tensor):
        """One collection step: f (N,) frames, r (N,) rewards, term (N,) bool."""
        g = self.frame_group[f.clamp(max=len(self.frame_group) - 1)]
        loss = (1 - (r / self.r_max).clamp(0, 1)) + self.beta * term.float()
        cnt = torch.zeros(self.G, device=self.device).index_add_(0, g, torch.ones_like(loss))
        mean = torch.zeros(self.G, device=self.device).index_add_(0, g, loss) / cnt.clamp(min=1)
        self.N.add_(cnt)
        run = cnt / self.N.clamp(min=1)  # running mean while N < 1 / alpha, then the EMA (cnt visits: 1 - (1 - a)^cnt)
        kf = torch.maximum(1 - (1 - self.af) ** cnt, run)
        ks = torch.maximum(1 - (1 - self.as_) ** cnt, run)
        self.fast.add_(kf * (mean - self.fast))
        self.slow.add_(ks * (mean - self.slow))
        self.v.mul_(0.999).add_(0.001 * cnt / cnt.sum().clamp(min=1))
        self.l_mean.mul_(0.999).add_(0.001 * loss.mean())

    @torch.no_grad()
    def tick(self, it: int):
        """Host side, outside the captured graphs, at the start of iteration it: snapshot and reference update."""
        every = int(self.snap_every)
        if it % every:
            return
        n = int((self.snap_t >= 0).sum())
        if n == self.snap_k:  # full: keep every other snapshot at twice the interval
            keep = torch.arange(0, self.snap_k, 2)
            self.ring_slow[: len(keep)] = self.ring_slow[keep.to(self.device)].clone()
            self.ring_N[: len(keep)] = self.ring_N[keep.to(self.device)].clone()
            self.snap_t[: len(keep)] = self.snap_t[keep].clone()
            self.snap_t[len(keep) :] = -1
            n, every = len(keep), every * 2
            self.snap_every.fill_(every)
            if it % every:
                return
        self.ring_slow[n].copy_(self.slow), self.ring_N[n].copy_(self.N)
        self.snap_t[n] = it
        t = self.snap_t[: n + 1]
        j = int(torch.argmin((t - it / 2).abs()))
        self.ref_slow.copy_(self.ring_slow[j]), self.ref_N.copy_(self.ring_N[j])
        self.gate_on.fill_(float(it - int(t[j]) >= self.min_window))

    # ---------------------------------------------------------------- allocation
    def _judge(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(deficit above the library's attainable level, net progress over the window, tried, judged) per group."""
        seen = self.N > 0
        loss = torch.where(seen, self.fast, self.l_mean)  # unseen groups: the library-average loss
        # attainable loss: the 5th percentile over the groups (the reward kernels never give a zero loss)
        floor = torch.sort(loss).values[int(0.05 * (self.G - 1))]
        d = (loss - floor).clamp(min=0)
        # progress of the slow average over the window, its noise measured on the library: groups that do not
        # progress scatter around 0 symmetrically, so the negative side gives the scale
        ok = self.ref_N > 0  # seen before the window: judged
        raw = torch.where(ok, self.ref_slow - self.slow, torch.zeros_like(d))
        se = (2 * (raw.clamp(max=0) ** 2).sum() / ok.sum().clamp(min=1)).sqrt().clamp(min=1e-4)
        lp = torch.minimum((raw - self.z * se).clamp(min=0) * ok, d)  # progress below the attainable level is moot
        dw = self.N - self.ref_N
        tried = ok & (dw >= self.tried_mult * (dw * ok).sum() / ok.sum().clamp(min=1))
        self.noise.copy_(se)
        return d, lp, tried, ok

    def score(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ranking score, stuck, net progress, deficit above the library's attainable level) per group."""
        d, lp, tried, _ = self._judge()
        stuck = tried & (lp <= 0) & (self.gate_on > 0)
        if not self.gate:
            stuck = torch.zeros_like(stuck)
        return torch.where(stuck, torch.zeros_like(d), d), stuck, lp, d

    def group_share(self) -> torch.Tensor:
        """Adaptive share per group (see the module docstring), capped at self.cap (the excess goes by the floor)."""
        main, _, _, _ = self.score()
        order = torch.argsort(main, descending=True)
        w = torch.zeros_like(main).scatter_(0, order, self.rank_w) * (main > 0)
        p = torch.where(w.sum() > 0, w / w.sum().clamp(min=1e-12), self.floor)
        for _ in range(3):  # cap: the excess goes to the uncapped groups by the floor (fixed passes, no host sync)
            over = p >= self.cap
            free = torch.where(over, torch.zeros_like(p), self.floor)
            excess = torch.where(over, p - self.cap, torch.zeros_like(p)).sum()
            p = torch.where(over, torch.full_like(p, self.cap), p + excess * free / free.sum().clamp(min=1e-12))
        self.target.copy_(p)
        return p

    def bin_probs(self) -> torch.Tensor:
        """The adaptive reset distribution over bins (uniform inside a group)."""
        p = self.group_share()
        return p[self.bin_group] / self.group_bins[self.bin_group]

    def record_mixed(self, bin_probs: torch.Tensor):
        """The full reset distribution of this collection step (after the floor and failure mixing)."""
        self.mixed.copy_(torch.zeros_like(self.mixed).index_add_(0, self.bin_group, bin_probs))
        self.a.mul_(self.a_decay).add_(self.mixed)

    def start_weight(self, f: torch.Tensor, wmax: float = 10.0) -> torch.Tensor:
        """Weights of candidate imagination starts at frames f: the group's reset share over its share of the data."""
        g = self.frame_group[f.clamp(max=len(self.frame_group) - 1)]
        return (self.mixed[g] / self.v[g].clamp(min=1e-9)).clamp(1 / wmax, wmax)

    # ---------------------------------------------------------------- diagnostics / checkpoints
    @torch.no_grad()
    def diag(self) -> dict:
        main, stuck, lp, d = self.score()
        p = self.target
        seen = self.N > 0
        mastered = seen & (d < 0.25 * d.mean())
        prog = seen & ~mastered & (lp > 0)  # lp is net of z noise levels
        noprog = seen & ~mastered & ~prog
        _, _, tried, _ = self._judge()
        top = torch.argsort(d, descending=True)[: max(1, self.G // 100)]
        ent = -(p * p.clamp(min=1e-12).log()).sum() / torch.log(torch.tensor(float(self.G)))
        return {"ent": float(ent), "mastered": float(p[mastered].sum()), "prog": float(p[prog].sum()),
                "noprog": float(p[noprog].sum()), "unseen": float(p[~seen].sum()), "top1": float(p[top].sum()),
                "n_mastered": int(mastered.sum()), "n_prog": int(prog.sum()), "n_noprog": int(noprog.sum()),
                "n_learn": int((main > 0).sum()), "n_tried": int(tried.sum()), "n_stuck": int(stuck.sum()),
                "noise": float(self.noise)}

    def state_dict(self) -> dict:
        return {k: getattr(self, k).cpu() for k in ("fast", "slow", "N", "a", "v", "l_mean", "noise", "target", "mixed",
                                                    "snap_every", "snap_t", "ring_slow", "ring_N", "ref_slow", "ref_N",
                                                    "gate_on")}  # fmt: skip

    def load_state_dict(self, sd: dict):
        for k, x in sd.items():
            getattr(self, k).copy_(x.to(getattr(self, k).device))

    def dump(self, path: str, names: list[str]):
        main, stuck, lp, d = self.score()
        torch.save({"group_clip": self.group_clip.cpu(), "group_bins": self.group_bins.cpu(), "names": names,
                    "main": main.cpu(), "stuck": stuck.cpu(), "lp": lp.cpu(), "d": d.cpu(), **self.state_dict()}, path)
