"""Clip-local sampling and timeout/reset regression checks without a simulator."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from partymimic.env import TrackingSim


def motion_fixture(lengths):
    lengths = torch.tensor(lengths)
    starts = lengths.cumsum(0) - lengths
    bin_clip, bin_start = [], []
    for c, length in enumerate(lengths.tolist()):
        for offset in range(0, length, 50):
            bin_clip.append(c)
            bin_start.append(int(starts[c]) + offset)
    return SimpleNamespace(
        clip_start=starts, clip_end=starts + lengths - 1,
        clip_id=torch.repeat_interleave(torch.arange(len(lengths)), lengths),
        bin_clip=torch.tensor(bin_clip), bin_start=torch.tensor(bin_start),
        num_bins=len(bin_start),
    )


class MotionBoundaryTests(unittest.TestCase):
    def test_every_bin_has_a_clip_local_next_frame(self):
        m = motion_fixture([2, 50, 51, 100, 101, 351, 51])
        env = TrackingSim.__new__(TrackingSim)
        env.motion, env.device = m, 'cpu'
        env.alloc, env.alloc_lambda, env.alloc_last = None, 0.0, False
        env.bin_G = torch.zeros(m.num_bins)
        env.bin_w_avg = torch.full((m.num_bins,), 1 / m.num_bins)
        env.G_max, env.div_p, env.fail_reset = 10., None, 0.
        bins = torch.arange(m.num_bins)
        # Visit every bin, including terminal-only bins in the middle of the
        # archive and at its final frame, with offsets near both endpoints.
        for gate in (None, torch.tensor(True), torch.tensor(False)):
            for offset in (0., 0.5, torch.nextafter(torch.tensor(1.), torch.tensor(0.)).item()):
                with self.subTest(gate=gate, offset=offset):
                    with patch('torch.multinomial', return_value=bins), \
                         patch('torch.rand', return_value=torch.full((m.num_bins,), offset)):
                        frames, selected = env.sample_frames(m.num_bins, gate=gate)
                    self.assertTrue(bool((frames >= m.clip_start[m.bin_clip[selected]]).all()))
                    self.assertTrue(bool((frames < m.clip_end[m.bin_clip[selected]]).all()))
                    torch.testing.assert_close(m.clip_id[frames], m.bin_clip[selected])
                    torch.testing.assert_close(m.clip_id[frames + 1], m.bin_clip[selected])


if __name__ == '__main__':
    unittest.main()
