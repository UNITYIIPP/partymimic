"""Convert train_log.csv to TensorBoard events without restarting training.

python tools/csv_tensorboard.py --log_dir logs/evt_s0_2gpu --follow
tensorboard --logdir logs/evt_s0_2gpu/tensorboard --port 6006
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter


GROUPS = {
    'wm_loss': 'world_model/loss', 'wm_err8': 'world_model/rollout_error_8',
    'pi_loss': 'policy/loss', 'pi_gradnorm': 'policy/gradient_norm',
    'v_loss': 'critic/loss', 'imag_rew': 'imagination/reward',
    'imag_survival': 'imagination/survival', 'sim_rew': 'rollout/reward',
    'ep_len': 'rollout/episode_length', 'ep_ret_norm': 'rollout/normalized_return',
    'term_frac': 'rollout/termination_fraction', 'it_per_s': 'performance/iterations_per_second',
    'sim_steps': 'performance/sim_steps', 'wall_h': 'performance/wall_hours',
    'lr': 'optimizer/base_lr', 'pi_lr': 'optimizer/policy_lr',
    'pi_mult': 'optimizer/policy_lr_multiplier', 'nonfinite': 'optimizer/nonfinite_updates',
}


def convert(source: Path, writer: SummaryWriter, last_step: int) -> tuple[int, int]:
    """Only consume complete rows: the trainer may be appending concurrently."""
    if not source.exists():
        return last_step, 0
    lines = source.read_text().splitlines(keepends=True)
    complete = [line for line in lines if line.endswith('\n')]
    added = 0
    for row in csv.DictReader(complete):
        try:
            step = int(row['iteration'])
            values = {key: float(value) for key, value in row.items() if key != 'iteration'}
        except (KeyError, TypeError, ValueError):
            continue
        if step <= last_step:
            continue
        for key, value in values.items():
            if math.isfinite(value):
                writer.add_scalar(GROUPS.get(key, f'train/{key}'), value, step)
        last_step, added = step, added + 1
    if added:
        writer.flush()
    return last_step, added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_dir', type=Path, required=True)
    parser.add_argument('--follow', action='store_true', help='keep importing new CSV rows')
    parser.add_argument('--interval', type=float, default=5.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    output = args.log_dir / 'tensorboard'
    # Resume conversion at the last imported iteration, avoiding duplicate points.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    last_step = -1
    if output.exists():
        acc = EventAccumulator(str(output), size_guidance={'scalars': 0})
        acc.Reload()
        last_step = max((event.step for tag in acc.Tags()['scalars'] for event in acc.Scalars(tag)), default=-1)
    try:
        with SummaryWriter(str(output)) as writer:
            while True:
                last_step, added = convert(args.log_dir / 'train_log.csv', writer, last_step)
                if added:
                    print(f'{added} rows imported; iteration {last_step}; events: {output}', flush=True)
                if not args.follow:
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
