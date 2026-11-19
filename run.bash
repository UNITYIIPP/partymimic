

#export motion
cd /mnt/dataset/lqi3lr/dyh/PartyMimic
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  bash scripts/export_motion.sh \
  logs/evt_lafan_seed19382_page_s0_1gpu/ckpt_100000.pt \
  /mnt/dataset/lqi3lr/dyh/allmimic/data/processed/evt_lafan1_50hz.npz \
  dance1_subject2 \
  logs/evt_lafan_seed19382_page_s0_1gpu/export_100000_dance1_subject2

cd /mnt/dataset/lqi3lr/dyh/PartyMimic

PM_RUN=logs/evt_lafan_seed19382_page_s0_1gpu
PM_OUT="$PM_RUN/export_100000_dance1_subject2"
PM_MOTION=/mnt/dataset/lqi3lr/dyh/allmimic/data/processed/evt_lafan1_50hz.npz

mkdir -p "$PM_OUT"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH="$PWD/src" \
  MOTION_FILE="$PM_MOTION" \
  .venv/bin/python tools/export_deploy_onnx.py \
  "$PM_RUN/ckpt_100000.pt" \
  "$PM_OUT/evt_policy_deploy.onnx"


python tools/export_deploy_onnx.py \
logs/evt_lafan_seed19382_evt_norminal/ckpt_024000.pt \
logs/evt_lafan_seed19382_evt_norminal/ckpt_024000/evt_policy_ckpt_024000.onnx