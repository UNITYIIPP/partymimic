#!/usr/bin/env bash
# Export a checkpoint with one baked-in motion and render its ONNX rollout.
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
  echo 'Usage: bash scripts/export_motion.sh <checkpoint.pt> <motion_archive.npz> <clip_name> <output_dir> [preview_seconds]' >&2
  exit 2
fi

pm_ckpt=$(realpath -e -- "$1")
pm_motion=$(realpath -e -- "$2")
pm_clip=$3
pm_out=$(realpath -m -- "$4")
pm_preview=()
if (( $# == 5 )); then
  pm_preview=(--seconds "$5")
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
pm_python=.venv/bin/python
"$pm_python" -c 'import onnx, onnxruntime'

mkdir -p "$pm_out"
pm_tmp=$(mktemp -d "$pm_out/.export_motion.XXXXXXXX")
trap 'for pm_file in "$pm_tmp"/policy.onnx "$pm_tmp"/rollout.npz "$pm_tmp"/track.mp4; do if [[ -f "$pm_file" ]]; then unlink "$pm_file"; fi; done; rmdir "$pm_tmp"' EXIT

CUDA_VISIBLE_DEVICES="" "$pm_python" tools/export_onnx.py \
  --ckpt "$pm_ckpt" --motion "$pm_motion" --clip "$pm_clip" --out "$pm_tmp/policy.onnx"

"$pm_python" tools/play_onnx.py "$pm_tmp/policy.onnx" \
  --motion "$pm_motion" --clip "$pm_clip" --record "$pm_tmp/rollout.npz" "${pm_preview[@]}"

pm_render_info=$("$pm_python" - "$pm_tmp/policy.onnx" "$pm_tmp/rollout.npz" <<'PY'
import sys

import numpy as np
import onnx

model = onnx.load(sys.argv[1], load_external_data=False)
metadata = {entry.key: entry.value for entry in model.metadata_props}
with np.load(sys.argv[2], allow_pickle=False) as rollout:
    frames = len(rollout["f"])
    if rollout["alive"][-1, 0] == 0:
        # Leave a short hold on the terminal frame so the viewer can see the termination label.
        frames += 25
seconds = frames * float(metadata["control_dt"])
print(metadata["robot"], f"{seconds + 1e-8:.8f}")
PY
)
read -r pm_robot pm_video_seconds <<< "$pm_render_info"
MUJOCO_GL=egl "$pm_python" tools/render.py "$pm_tmp/rollout.npz" "$pm_tmp/track.mp4" \
  --robot "$pm_robot" --title "$(basename "${pm_ckpt%.pt}") ONNX" --seconds "$pm_video_seconds" --fps 25

# Publish the verified export and video once both have been generated.
exec {pm_output_lock}<"$pm_out"
flock "$pm_output_lock"
mv -- "$pm_tmp/policy.onnx" "$pm_tmp/rollout.npz" "$pm_tmp/track.mp4" "$pm_out/"
flock -u "$pm_output_lock"
printf 'ONNX: %s\nVideo: %s\n' "$pm_out/policy.onnx" "$pm_out/track.mp4"
