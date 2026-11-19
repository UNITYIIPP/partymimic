# EVT 训练与模型导出

上游代码版本为 `Roboparty/PartyMimic@7692a7292f21af8a1b732fa2abdab31d6ecd01ac`。`evt` 和 `evt_nominal` 各自使用原生关节名称和控制参数，仿真足部接触判定高度均为 0.07 m。`--motion_file` 将 `compact_v1` 归档解包为独立片段，再交给上游的重采样、分箱、重置和 replay 流程。以下命令从仓库根目录执行。

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4
ROBOT=evt
MOTION=/mnt/dataset/lqi3lr/dyh/allmimic/data/processed/evt_lafan1_50hz.npz
PRIOR="data/${ROBOT}_lafan1_prior.npz"
RUN="logs/${ROBOT}_official_page_s0"
```

当前工作区已备好两套机器人资产。如果需要重建所选机器人的 XML 和 mesh，运行下面的命令。

```bash
.venv/bin/python tools/prepare_evt.py --robot "$ROBOT" \
  --source /mnt/dataset/lqi3lr/dyh/allmimic/assets/robots/evt
```

## 使用官方主页训练配方

对每个动作库和机器人生成一次先验。当前工作区已生成 `data/evt_lafan1_prior.npz`。本例的 40 个片段使用 24 个动作簇。`alloc.py` 和 `spot.py` 采用上游实现，双卡时 learner 通过现有 pipeline 消息接收分配器的分组重置份额和访问频率。

```bash
.venv/bin/python tools/motion_prior.py "$MOTION" "$PRIOR" --robot "$ROBOT"

CUDA_VISIBLE_DEVICES=1,2 .venv/bin/python -m partymimic.train \
  --robot "$ROBOT" --motion_file "$MOTION" --log_dir "$RUN" --seed 0 --pipeline \
  --std_scale 0.35 --obs_err kp --wrist_kernel 0.5 --wrist_payload 2 \
  --reset_alloc lp --start_alloc lp --alloc_lambda 0.15 --alloc_last 1 \
  --cover_clusters 24 --cover_win 100000000 --motion_prior "$PRIOR" \
  --v_clip 10 --v_clip_raw --wm_guard 5 --v_guard 5 --action_noise 0
```

单卡运行时选择一个 GPU 并去掉 `--pipeline`。如需官方默认配方，可用另一运行目录启动以下命令。

```bash
CUDA_VISIBLE_DEVICES=1,2 .venv/bin/python -m partymimic.train \
  --robot "$ROBOT" --motion_file "$MOTION" --pipeline --seed 0 \
  --log_dir "logs/${ROBOT}_official_default_s0"
```

## 导出部署模型

`RUN` 指向要导出的实际运行目录。训练默认结束于第 12000 次迭代，以下命令读取相应 checkpoint，输出动作参考作为运行时输入的策略。

```bash
.venv/bin/python tools/export_deploy_onnx.py \
  "$RUN/ckpt_012000.pt" "$RUN/${ROBOT}_policy.onnx"
```

同目录会生成 `.onnx`、`.yaml` 和 `.check.npz`。后者保存一组输入和 PyTorch 的参考动作，便于验证推理引擎。动作归档搬家后可用 `MOTION_FILE=/新的路径/动作.npz` 指定导出校验用的动作。切换到 `evt_nominal` 时，在开头设定 `ROBOT=evt_nominal`，并重新设置 `PRIOR` 和 `RUN`。
