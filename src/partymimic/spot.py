"""Cluster allocation of the resets (--alloc_kind cover; alloc.Allocator keeps the per-group statistics).

Cover: units are k-means clusters of per-clip motion statistics (root height, uprightness, root / joint speeds; copies
"__cN" of a clip are one unit). Every `period` iterations the top `m` units by deficit x prior share the adaptive part
of the resets in proportion to that score, each capped at `cap` of it; a unit keeps at least `maint` x its slowly
decaying peak share and loses it after holding share for `win` iterations without beating the library's relative loss
drop over them. The prior (--motion_prior, tools/motion_prior.py) marks the clips that are physically feasible in the
simulator (no support by an object it does not have) and weights units by their root angular speed: a fixed share of
the resets helps feasible dynamic clips and not the clips held up by a missing object, whatever their error.

Spotlight is the base class (units, clustering, unit losses). All decisions are made on the host in tick(), outside
the captured graphs; the graphs read the share table it leaves in place (group_share / bin_probs / start_weight of
Allocator)."""

from __future__ import annotations

import re

import torch

from partymimic.alloc import Allocator

IDLE, PROBE, COMMIT, COOL, DONE = 0, 1, 2, 3, 4


@torch.no_grad()
def clip_features(m) -> torch.Tensor:
    """(clips, 9) standardized statistics of each clip's reference motion (host)."""
    c, C = m.clip_id, len(m.names)
    n = m.clip_len.float()
    mean = lambda x: torch.zeros(C, device=x.device).index_add_(0, c, x) / n  # noqa: E731
    amin = lambda x: torch.full((C,), float("inf"), device=x.device).index_reduce_(0, c, x, "amin")  # noqa: E731
    z, gz = m.root_p[:, 2], m.root_gz
    w, v = m.root_w.norm(dim=-1), m.root_v[:, :2].norm(dim=-1)
    zm = mean(z)
    f = torch.stack((zm, amin(z), (mean(z * z) - zm * zm).clamp(min=0).sqrt(), amin(gz), mean(gz), mean(w),
                     mean(w * w).sqrt(), mean((m.qd * m.qd).mean(-1)).sqrt(), mean(v)), 1).float().cpu()
    return ((f - f.mean(0)) / f.std(0).clamp(min=1e-6)).clamp(-5, 5)


def kmeans(x: torch.Tensor, k: int, gen: torch.Generator, iters: int = 50) -> torch.Tensor:
    """Cluster index per row (k-means++ initialization)."""
    cen = x[torch.randint(len(x), (1,), generator=gen)]
    for _ in range(k - 1):
        d = torch.cdist(x, cen).min(1).values ** 2
        cen = torch.cat((cen, x[torch.multinomial(d / d.sum(), 1, generator=gen)]))
    for _ in range(iters):
        a = torch.cdist(x, cen).argmin(1)
        cnt = torch.bincount(a, minlength=k).float()
        new = torch.zeros_like(cen).index_add_(0, a, x) / cnt.clamp(min=1).unsqueeze(1)
        cen = torch.where(cnt.unsqueeze(1) > 0, new, cen)
    return torch.cdist(x, cen).argmin(1)


class Spotlight(Allocator):
    def __init__(self, motion, device: str, names: list[str], k: int = 48, warm: int = 150, probe: int = 600,
                 rho: float = 0.1, min_probe: int = 16, period: int = 150, seed: int = 0, units: str = "clip",
                 n_clusters: int = 128, share: str = "equal", unit_cap: float = 0.5, **kw):  # fmt: skip
        super().__init__(motion, device, **kw)
        self.k, self.warm, self.probe, self.rho, self.min_probe, self.period = k, warm, probe, rho, min_probe, period
        self.share, self.unit_cap = share, unit_cap
        self.gen = torch.Generator().manual_seed(seed)
        base = [re.sub(r"__c\d+$", "", n) for n in names]
        uid = {b: i for i, b in enumerate(dict.fromkeys(base))}
        clip_unit = torch.tensor([uid[b] for b in base])
        self.unit_names = list(uid)
        if units == "cluster":  # copies of a clip have its features: cluster the distinct clips
            pos = {}
            for i, b in enumerate(base):
                pos.setdefault(b, i)
            first = torch.tensor([pos[b] for b in self.unit_names])
            cl = kmeans(clip_features(motion)[first], n_clusters, self.gen)
            used = torch.unique(cl)
            cl = torch.searchsorted(used, cl)
            clip_unit = cl[clip_unit]
            self.unit_names = []
            for j in range(len(used)):
                mem = [self.unit_names_of(b) for b, u in zip(uid, cl.tolist()) if u == j]
                top = sorted(set(mem), key=mem.count, reverse=True)[:3]
                self.unit_names.append(f"k{j}:{len(mem)}:" + ",".join(f"{t}x{mem.count(t)}" for t in top))
        self.clip_unit = clip_unit
        self.U = len(self.unit_names)
        self.group_unit = clip_unit.to(device)[self.group_clip]
        gu = self.group_unit.cpu()
        self.unit_bins = torch.zeros(self.U).index_add_(0, gu, self.group_bins.cpu())
        self.status = torch.zeros(self.U, dtype=torch.long)  # host state
        self.t0 = torch.zeros(self.U, dtype=torch.long)  # iteration the unit entered its slot / was last judged
        self.l0 = torch.full((self.U,), float("nan"))  # loss at the start of the judged period
        self.b0 = torch.full((self.U,), float("nan"))  # library loss (l_mean) at the same time
        self.n_probed = torch.zeros((), dtype=torch.long)
        self.target.copy_(self.floor)  # until the first slots are filled

    @staticmethod
    def unit_names_of(name: str) -> str:
        """Family of a clip name: its first two words without digits."""
        return "_".join([w for w in name.split("_") if w and not any(ch.isdigit() for ch in w)][:2])

    def _unit_loss(self) -> torch.Tensor:
        gu = self.group_unit.cpu()
        w = self.group_bins.cpu()
        f = self.fast.cpu()
        return torch.zeros(self.U).index_add_(0, gu, f * w) / self.unit_bins

    def _unit_deficit(self) -> torch.Tensor:
        d, _, _, _ = self._judge()
        gu = self.group_unit.cpu()
        w = self.group_bins.cpu()
        seen = torch.zeros(self.U).index_add_(0, gu, (self.N.cpu() > 0).float() * w) / self.unit_bins
        return torch.where(seen >= 0.5, torch.zeros(self.U).index_add_(0, gu, d.cpu() * w) / self.unit_bins,
                           torch.full((self.U,), -1.0))  # units mostly unseen are not candidates yet

    @torch.no_grad()
    def tick(self, it: int):
        super().tick(it)  # keeps the v4 window statistics (diagnostics, dumps)
        if it % self.period or it == 0:
            return
        l = self._unit_loss()
        st = self.status
        # probes: read l0 after the warm-up, judge after `probe` more iterations
        warm_done = (st == PROBE) & torch.isnan(self.l0) & (it - self.t0 >= self.warm)
        lib = float(self.l_mean)
        self.l0[warm_done], self.b0[warm_done] = l[warm_done], lib
        self.t0[warm_done] = it
        judged = (st == PROBE) & ~torch.isnan(self.l0) & (it - self.t0 >= self.probe)
        # relative drop beyond the library's over the same period (early in training everything improves)
        drop = (self.l0 - l) / self.l0.clamp(min=1e-6) - (self.b0 - lib) / self.b0.clamp(min=1e-6)
        st[judged & (drop >= self.rho)] = COMMIT
        st[judged & (drop < self.rho)] = COOL
        # committed: re-judged every `probe` iterations
        rej = (st == COMMIT) & (it - self.t0 >= self.probe) & ~judged
        st[rej & (drop < self.rho / 2)] = DONE
        renew = (judged & (st == COMMIT)) | (rej & (st == COMMIT))
        self.l0[renew], self.b0[renew], self.t0[renew] = l[renew], lib, it
        # fill free slots with new probes, drawn in proportion to the deficit (the largest deficits are mostly
        # clips that cannot be learned: strict largest-first would spend every slot on them)
        n_commit = int((st == COMMIT).sum())
        if n_commit > self.k - self.min_probe:  # too many committed: the oldest ones graduate
            old = torch.nonzero(st == COMMIT).flatten()
            old = old[torch.argsort(self.t0[old])][: n_commit - (self.k - self.min_probe)]
            st[old] = DONE
        free = self.k - int(((st == PROBE) | (st == COMMIT)).sum())
        if free > 0:
            d = self._unit_deficit().clamp(min=0)
            d[st != IDLE] = 0.0
            n = min(free, int((d > 0).sum()))
            cand = torch.multinomial(d, n, replacement=False, generator=self.gen) if n > 0 else torch.zeros(0, dtype=torch.long)
            st[cand] = PROBE
            self.t0[cand] = it
            self.l0[cand] = float("nan")
            self.n_probed += len(cand)
        # share table: slots equally, uniform over a unit's bins
        slot = ((st == PROBE) | (st == COMMIT)).float()
        if self.share == "boost":  # the same boost over every unit's natural share, each unit capped
            w = slot * self.unit_bins
            unit_p = torch.minimum(w / w.sum().clamp(min=1e-9), torch.tensor(self.unit_cap))
        else:
            unit_p = slot
        unit_p = unit_p / unit_p.sum().clamp(min=1e-9)
        gp = unit_p[self.group_unit.cpu()] * self.group_w / self.unit_w[self.group_unit.cpu()].clamp(min=1e-9)
        self.target.copy_(gp.to(self.device) if slot.sum() > 0 else self.floor)

    def group_share(self) -> torch.Tensor:
        return self.target

    @torch.no_grad()
    def diag(self) -> dict:
        st = self.status
        p = self.target
        ent = -(p * p.clamp(min=1e-12).log()).sum() / torch.log(torch.tensor(float(self.G)))
        return {"ent": float(ent), "mastered": 0.0, "prog": 0.0, "noprog": 0.0, "unseen": 0.0, "top1": 0.0,
                "n_mastered": int((st == DONE).sum()), "n_prog": int((st == PROBE).sum()),
                "n_noprog": int((st == COOL).sum()), "n_learn": int((st == COMMIT).sum()),
                "n_tried": int(self.n_probed), "n_stuck": int((st == COOL).sum()), "noise": float(self.noise)}

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd.update(status=self.status.clone(), t0=self.t0.clone(), l0=self.l0.clone(), b0=self.b0.clone(),
                  n_probed=self.n_probed.clone())
        return sd

    def dump(self, path: str, names: list[str]):
        super().dump(path, names)
        d = torch.load(path)
        d.update(unit_names=self.unit_names, clip_unit=self.clip_unit, group_unit=self.group_unit.cpu(), status=self.status.clone(),
                 t0=self.t0.clone(), l0=self.l0.clone(), unit_loss=self._unit_loss())
        torch.save(d, path)


class Cover(Spotlight):
    """v6: coverage. In v5 the Extreme clips that got share reached the oracle's error and the others stayed at the
    uniform one (k = 2 clusters held 2-5 of the 14 clips; the clips spread over 4-8 clusters); a clip whose share
    went back to 0 lost what it had learned within 12k iterations. Units are k-means clusters of the clips (or clips,
    copies "__cN" being one unit; by deficit x prior the Extreme clips rank 50-1400 among ~800 similar kungfu clips,
    so a clip-level top-m rarely reaches them). Every `period` iterations from `start` on (default 0: the prior needs no probe), the top `m` feasible units by
    deficit x prior share the adaptive part in proportion to the score, each capped at `cap` of it (the excess goes
    to the others); a unit keeps at least `maint` x its (slowly decaying) peak share, and loses it only after holding
    share for `win` iterations without beating the library's relative loss drop over the last `win` iterations.
    Unseen units count the average deficit. Prior = feasible (unsup < max_unsup) x mean 95th-percentile root angular
    speed over the unit's feasible bins x their share of its bins, scaled to max 1 (use_prior False: feasible share only).
    Inside a unit the share goes by feasible bins."""

    def __init__(self, motion, device: str, names: list[str], prior: str = "", m: int = 32, cap: float = 0.05,
                 maint: float = 0.5, decay: float = 0.995, win: int = 3000, start: int = 0, period: int = 250,
                 use_prior: bool = True, max_unsup: float = 0.1, seed: int = 0, units: str = "cluster",
                 n_clusters: int = 128, **kw):  # fmt: skip
        super().__init__(motion, device, names, k=m, period=period, seed=seed, units=units, n_clusters=n_clusters, **kw)
        self.m, self.cap, self.maint, self.decay, self.win, self.start = m, cap, maint, decay, win, start
        C = len(names)
        feas, ang = torch.ones(C), torch.ones(C)
        self.n_missing = 0
        if prior:
            import numpy as np

            z = np.load(prior)
            cols = list(z["cols"])
            base = lambda n: re.sub(r"__c\d+$", "", n)  # noqa: E731
            feats = {}
            for n, row in zip(z["names"].tolist(), z["x"]):
                feats.setdefault(base(n), (float(row[cols.index("unsup")]), float(row[cols.index("angvel")])))
            known = [feats[base(n)] for n in names if base(n) in feats]
            med = float(np.median([a for _, a in known]))
            f = torch.tensor([feats.get(base(n), (0.0, med)) for n in names])
            self.n_missing = C - len(known)
            feas, ang = (f[:, 0] < max_unsup).float(), f[:, 1]
        gc, gu = self.group_clip.cpu(), self.group_unit.cpu()
        self.group_w = self.group_bins.cpu() * feas[gc]
        self.unit_w = torch.zeros(self.U).index_add_(0, gu, self.group_w)
        unit_ang = torch.zeros(self.U).index_add_(0, gu, self.group_w * ang[gc]) / self.unit_w.clamp(min=1e-9)
        p = self.unit_w / self.unit_bins.clamp(min=1e-9) * (unit_ang if use_prior else 1.0)
        self.prior = p / p.max().clamp(min=1e-9)
        self.feasible_share = float(self.group_w.sum() / self.group_bins.cpu().sum())
        self.share = torch.zeros(self.U)
        self.peak = torch.zeros(self.U)
        self.cool = torch.zeros(self.U, dtype=torch.bool)
        self.on_t = torch.full((self.U,), -1, dtype=torch.long)  # iteration the unit got share, -1 none
        self.hist_t: list[int] = []
        self.hist_l: list[torch.Tensor] = []
        self.hist_lib: list[float] = []

    def _deficit(self) -> torch.Tensor:
        """Unit deficit over the feasible bins, relative to the mean over the units mostly seen (unseen: 1)."""
        d, _, _, _ = self._judge()
        gu, w = self.group_unit.cpu(), self.group_w
        seen = (self.N.cpu() > 0).float()
        num = torch.zeros(self.U).index_add_(0, gu, d.cpu() * w * seen)
        den = torch.zeros(self.U).index_add_(0, gu, w * seen)
        has = den >= 0.5 * self.unit_w.clamp(min=1e-9)
        du = num / den.clamp(min=1e-9)
        ref = float(du[has].mean()) if bool(has.any()) and float(du[has].mean()) > 0 else 1.0
        return torch.where(has, du / ref, torch.ones(self.U))

    @torch.no_grad()
    def tick(self, it: int):
        Allocator.tick(self, it)
        if it % self.period or it < self.start:
            return
        l, lib = self._unit_loss(), float(self.l_mean)
        self.hist_t.append(it), self.hist_l.append(l.clone()), self.hist_lib.append(lib)
        j = min(range(len(self.hist_t)), key=lambda i: abs(self.hist_t[i] - (it - self.win)))
        if it - self.hist_t[j] >= 0.9 * self.win:
            l0, lib0 = self.hist_l[j], self.hist_lib[j]
            rel = (l0 - l) / l0.clamp(min=1e-6) - (lib0 - lib) / max(lib0, 1e-6)
            self.cool |= (self.on_t >= 0) & (it - self.on_t >= self.win) & (rel <= 0)
        score = (self._deficit() * self.prior).clamp(min=0)
        score[self.cool] = 0
        top = score.argsort(descending=True)[: self.m]
        w = torch.zeros(self.U)
        w[top] = score[top]
        s = w / w.sum().clamp(min=1e-12)
        for _ in range(20):  # cap: the excess goes to the uncapped candidates by score
            over = s > self.cap
            if not bool(over.any()):
                break
            excess = float((s[over] - self.cap).sum())
            s[over] = self.cap
            free = (w > 0) & (s < self.cap)
            if not bool(free.any()):
                break
            s[free] += excess * w[free] / w[free].sum()
        self.peak = torch.maximum(self.peak * self.decay, s)
        self.peak[self.cool] = 0
        s = torch.maximum(s, self.maint * self.peak)
        s = s / s.sum().clamp(min=1.0)  # the maintained shares may sum above 1
        self.on_t[(s > 0) & (self.on_t < 0)] = it
        self.on_t[s <= 0] = -1
        self.share = s
        gu = self.group_unit.cpu()
        gp = s[gu] * self.group_w / self.unit_w[gu].clamp(min=1e-9)
        rest = 1.0 - float(gp.sum())  # all candidates capped: the rest by the floor
        self.target.copy_((gp + max(rest, 0.0) * self.floor.cpu()).to(self.device) if float(s.sum()) > 0 else self.floor)

    @torch.no_grad()
    def diag(self) -> dict:
        p = self.target
        ent = -(p * p.clamp(min=1e-12).log()).sum() / torch.log(torch.tensor(float(self.G)))
        return {"ent": float(ent), "mastered": 0.0, "prog": 0.0, "noprog": 0.0, "unseen": 0.0, "top1": float(self.share.max()),
                "n_mastered": 0, "n_prog": 0, "n_noprog": 0, "n_learn": int((self.share > 0).sum()),
                "n_tried": int((self.on_t >= 0).sum() + self.cool.sum()), "n_stuck": int(self.cool.sum()), "noise": float(self.noise)}

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd.update(share=self.share.clone(), peak=self.peak.clone(), cool=self.cool.clone(), on_t=self.on_t.clone())
        return sd

    def dump(self, path: str, names: list[str]):
        super().dump(path, names)
        d = torch.load(path)
        d.update(share=self.share.clone(), peak=self.peak.clone(), cool=self.cool.clone(), on_t=self.on_t.clone(),
                 prior=self.prior.clone())
        torch.save(d, path)
