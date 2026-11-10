**PartyMimic 与 allmimic 代码综合对比（2026-11-10）**

对比对象为本地 PartyMimic `536be97` 和 `/mnt/dataset/lqi3lr/dyh/allmimic` `01e6ad2` 的当前代码。主线分别是 `partymimic.train` 和 `allmimic.train`。这里比较代码实际默认值及调用路径，不把 README 的不同数据集成绩当成同条件实验。

PartyMimic 默认机器人是 `g1mj`，另有 `g1` 和 28 自由度 `rp1`；allmimic 默认 `g1`，另有 29 自由度 `evt`。IDE 中的 `src/partymimic/rp1/__init__.py` 是空文件，RP1 参数在 [rp1_constants.py](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/rp1/rp1_constants.py:28)。

allmimic 必须分四种配置看：裸默认、`recommended`、`root-feedback`、`root-feedback-delay`。后两者逐级继承推荐配置，分别增加根参考反馈和动作延迟。下文未另说明时，“推荐配置”指 `recommended`，并不默认包含另外两个预设。

**1. 主要结论**

两边都是 SuperMimic / SHAC 风格的基于模型学习：真实仿真采集 → replay → 拟合可微世界模型 → 在冻结的世界模型中展开策略 → 用解析奖励、软终止和末端价值反向传播 → 拟合 critic。主训练入口都不是 PPO；allmimic 另外提供 PPO-Sim 和 PPO-WM 基线。

| 方面 | PartyMimic 默认 | allmimic 推荐配置 |
|---|---|---|
| 世界模型 | 小型、带 LayerNorm 的结构化动力学 MLP | 较大 MLP，直接预测状态增量 |
| 策略 | MLP 编码器，32 维 latent；可选交叉注意力 | MLP 编码器，128 维 latent |
| 参考动作 | 全身以下一帧参考关节为基准，加网络修正 | 上身参考残差，下肢相对默认姿态的绝对动作 |
| 根姿态反馈 | actor 默认包含当前参考相对根姿态 | 推荐配置不包含；`root-feedback` 才增加 |
| 训练奖励 | 更窄的位置/姿态核，想象阶段增加接触与防滑 | 原始宽度，无接触项 |
| 优化 | policy / WM 默认 Muon + Adam；策略 EMA、SIL、价值梯度一致性 | 默认 Adam；可选 Muon；推荐配置有 value clip |
| 采集动作 | 全部环境加探索噪声 | 25% 环境不加探索动作噪声 |
| 多样性重置 | 启动时自动聚为 24 类，默认每类同等质量 | 需要外部 bin-prior 文件；工具默认 256 类，推荐指数 0.3 |
| 随机化 | 摩擦和 CoM 每环境初始化一次，编码器偏移每回合 | 摩擦、CoM、编码器偏移每回合重采样 |
| 双 GPU | 采集/WM 与 policy/critic 流水并行 | 数据并行，每卡采集、训练后同步梯度 |
| 主要工程取向 | 缩小 WM、图捕获、较短训练配方、多参考接口 | 长训练稳定性、任务/动力学开关、EVT 和延迟部署链路 |

这些是实现取向，不足以单独证明哪边在相同机器人和数据上效果更好。

**2. 网络结构、维度与参数量**

来源：[PartyMimic networks.py](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/networks.py:109)、[PartyMimic 网络实例化](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/train.py:184)、[allmimic networks.py](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/networks.py:103)、[allmimic 网络实例化](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/train.py:264)。

| 网络配置 | PartyMimic | allmimic 裸默认 | allmimic recommended |
|---|---|---|---|
| WM 隐藏层 | 1024 → 1024 → 1024 | 2048 → 2048 → 2048 | 同裸默认 |
| WM 激活 | SiLU | SiLU | SiLU |
| WM LayerNorm | 每个隐藏层有 | 无 | 无 |
| WM 输出层初始化 | 权重、偏置为零 | 权重正交初始化 gain=0.01，偏置零 | 同裸默认 |
| 编码器隐藏层 | 2048 → 1024 → 512 → 256 | 相同 | 相同 |
| latent | 32，tanh | 32，tanh | 128，tanh |
| 解码器隐藏层 | 2048 → 1024 → 512 → 256 | 相同 | 相同 |
| 编码器/解码器激活 | ELU | ELU | ELU |
| critic | 256 → 128 → 1，ELU | 相同 | 相同 |
| actor / WM 优化器 | 分别可选，默认 Muon + Adam | Adam | Adam |
| critic 优化器 | FlatAdam | fused Adam | fused Adam |

PartyMimic 的 `WorldModel` 类构造器隐藏层默认仍写着 2048×3，但 **train.py 显式传入 1024×3**，应以实际入口为准。allmimic 的 `Policy` 注释写 32 维，但推荐预设实际传入 128。

下表为可训练参数数目，不包括归一化 buffers、优化器状态、EMA 副本；均为 joint 接口、MLP、单 critic。PartyMimic 使用独立抽取的网络类在 CPU 实例化核对，allmimic 直接实例化核对。

| 配置 | WM 输入 → 输出 | actor 历史 / goal | critic 输入 | WM 参数 | actor 参数 | critic 参数 |
|---|---|---|---:|---:|---:|---:|
| PartyMimic G1 | 133 → 70 | 990 / 730 | 914 | 2,314,310 | 9,116,477 | 267,265 |
| PartyMimic RP1 | 129 → 68 | 960 / 710 | 890 | 2,308,164 | 9,013,820 | 261,121 |
| allmimic G1/EVT 裸默认 | 307 → 70 | 930 / 730 | 1009 | 9,166,918 | 8,993,597 | 291,585 |
| allmimic recommended | 307 → 70 | 930 / 730 | 1009 | 9,166,918 | 9,214,877 | 291,585 |
| allmimic root-feedback | 307 → 70 | 990 / 730 | 1099 | 9,166,918 | 9,337,757 | 314,625 |
| allmimic root-feedback-delay | 340 → 70 | 990 / 730 | 1132 | 9,234,502 | 9,337,757 | 323,073 |

allmimic 的 G1、EVT 编译模型均为根体 + 29 个运动链 body，状态特征为 `71 + 29×6 = 245`。延迟开启后 xi 从 33 增为 66，所以 WM 输入从 307 增为 340。

PartyMimic G1 的 WM 参数约为 allmimic 的 25.2%，但 actor 大小很接近；不能理解为所有网络都缩小了四倍。按上述默认计算，WM+actor+critic 合计约 11.70M 对 18.67M。

PartyMimic 可选 `--ref_enc xattn`：128 维、4 heads，历史观测经过一个因果自注意力块，用最新 token 查询未来参考帧，再经交叉注意力块输出 latent；这时 latent 不经过 MLP 路径的 tanh。allmimic 主策略没有这一结构。

**3. Actor 观测与参考指令**

来源：[PartyMimic 观测](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/mdp.py:110)、[allmimic 观测](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/task.py:218)。设 J 为关节数，历史长度默认 10。

| 单帧 actor 观测 | PartyMimic | allmimic |
|---|---|---|
| 关节位置 | q − q0 − offset，J 维 | q − q0 + bias，29 维 |
| 关节速度 | J 维 | 29 维 |
| 根体局部角速度 | 3 维 | 3 维 |
| 投影重力 | 3 维 | 3 维 |
| 上一步动作 | J 维；含参考 skip 后的动作 | 29 维；actor 原始输出，推荐配置上身为残差 |
| 当前参考相对根姿态 | 默认有，6 维，R_robotᵀ R_ref | 默认/推荐无；root-feedback 有 |
| 单帧总维度 | 3J+12：G1=99，RP1=96 | 默认/推荐=93；root-feedback=99 |
| 10 帧堆叠 | G1=990，RP1=960 | 默认/推荐=930；root-feedback=990 |
| 是否直接输入全局位置/根线速度 | 均没有 | 均没有 |

两边的编码器偏移符号约定相反，但各自观测与 PD target 一致，且分布关于零对称；不能仅凭正负号认定实现错误。

**特别注意 6D 旋转的排列不一致。** PartyMimic 取 `R[..., :, :2].reshape(...)`，顺序为 `[R00,R01,R10,R11,R20,R21]`；allmimic 先转置这两列，顺序为 `[R00,R10,R20,R01,R11,R21]`。转换索引为 `[0,2,4,1,3,5]`（Party → allmimic）。即使两边观测都是 99 维，也不能直接互换 ONNX 输入。

来源：[PartyMimic mat_to_6d](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/rotations.py:129)、[allmimic rot6d](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/rotations.py:130)。

参考指令默认取未来第 1～10 帧，50 Hz，即未来 20～200 ms；位置相对当前参考根位置，姿态相对当前参考 heading，并非相对机器人实际全局位置。

| 参考特征 | PartyMimic joint | allmimic current_ref |
|---|---|---|
| 每帧顺序 | Δp(3), R6(6), v_local(3), w_local(3), q(J), 0.1·qd(J) | Δp(3), R6(6), q(29), qd(29), v_local(3), w_local(3) |
| 每帧维度 | 15+2J；G1=73、RP1=71 | 73 |
| 总 goal | G1=730、RP1=710 | 730 |
| 可选参考坐标 | 当前发布入口固定参考自身坐标 | current_ref / robot / ref |
| 可选时间偏移 | 当前入口连续 1～K | 可配置稀疏 `future_offsets`，例如最远 60 帧=1.2 s |

来源：[PartyMimic goal](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/motion.py:168)、[allmimic goal](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/task.py:258)。

PartyMimic 还提供 `sparse5`：每参考帧 51 维，由 pelvis 位姿9、头部姿态6、左右腕/踝位姿各9组成，10帧为510维；没有参考关节角/速度。actor 追加27维关键点误差，解码器只用最新一帧的这27维。默认预训练冻结的 `kp2q`（1024/1024/512，20,000步，batch8192），恢复参考动作 skip；另有权重1.0的关节预测辅助损失。这里的“头”用 torso 对应刚体代表。allmimic 主入口无对应接口。

归一化也不同：PartyMimic 的 obs RunningNorm 在单帧维度上共享历史时刻统计，累计约2e8样本后冻结，不做±10裁剪；allmimic 对摊平后的整段历史归一化，各历史槽位有独立统计，默认持续更新并裁剪至±10。PartyMimic 虽然定义 `obs_scale()`，当前训练调用链没有乘它；不要把该函数中的 qd×0.1、gyro×0.25 当成已应用的观测缩放。goal 内的 qd×0.1 则确实应用。

**4. 世界模型与 critic 的特权信息**

PartyMimic WM 输入为：高度1 + 去heading根姿态6 + 根体坐标线速度3/角速度3 + q/qd 各J + 动作J + 归一化xi(J+4)。默认不需要 FK body 特征。若机器人配置了动作延迟，类本身可以另接前一步动作，但这不代表当前训练完整支持延迟。

allmimic WM 输入为：根状态71维 + 除根外29个body的位置/线速度174维 + 绝对setpoint动作29维 + xi33维。根线速度在heading系，角速度在根体系；位置特征使用 q−q0。两边都学习确定性动力学，奖励、观测和终止由可微解析函数生成，不由独立神经头预测。

PartyMimic 默认 `wm_structured=True`，网络学习速度增量及位置/姿态修正。积分使用旧/新速度权重0.375/0.625、dt=0.02，再加预测修正；allmimic 默认直接学习位置、旋转、速度、关节的增量，没有此结构。allmimic 可选 `wm_rot_out=tripoint`，将旋转输出由3维轴角改为9维标记点，输出70→76。

| WM 训练 | PartyMimic | allmimic recommended |
|---|---|---|
| 自回归窗口 | 8 步 | 8 步 |
| 每次学习阶段更新次数 | 5 | 10 |
| batch | 2048 | 2048 |
| 状态误差 | 按增量标准差归一化，额外逐类最小尺度 | 按增量标准差归一化 |
| 逐关节误差权重 | 默认均匀 | tracked-body误差传播权重；先裁至[0.25,4]再归一化 |
| 左右镜像数据增强 | 默认50% WM batch | 主训练无对应增强 |
| WM 输入标准差 eps | 0.01 | 0.001，可配置 |

PartyMimic 误差最小尺度按位置、旋转、线速度、角速度、关节角、关节速度为 `0.003, 0.005, 0.02, 0.05, 0.003, 0.05`。来源：[WM loss](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/train.py:305)、[allmimic WM loss](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/train.py:504)。

PartyMimic critic：紧凑状态 + 当前关节误差 + 14个跟踪body的位置误差42维 + 根位置误差3 + 相对姿态6 + xi + goal，G1共914维。allmimic 默认 critic：245维body增强状态 + goal730 + xi33 + 剩余帧数标量1，共1009维；root-feedback 再增加90维未来参考根位姿相对当前机器人，成为1099维。

因此 PartyMimic critic 默认有**当前**参考跟踪误差，allmimic root-feedback 增加的是**未来多帧**根误差，二者不是完全相同的输入设计。

**5. 动作、控制与奖励语义**

令 u 为解码器输出，α为每关节动作缩放，忽略编码器偏移：

| 模式 | 真实 PD 目标 |
|---|---|
| PartyMimic 默认 | q_target = q_ref(t+1) + α·u，全身残差 |
| PartyMimic 关闭 pi_ref_skip | q_target = q0 + α·u |
| allmimic 裸默认 | q_target = q0 + α·u |
| allmimic recommended 下肢 | q_target = q0 + α·u |
| allmimic recommended 上身 | q_target = q_ref(t+1) + res_scale·α·u，默认res_scale=1 |

关键差异是 skip 放置位置：PartyMimic 在 policy 内相加，policy 返回的 a 已含参考；allmimic 在 `Task.setpoint_action()` 外部把原始输出转为绝对setpoint，WM/replay使用setpoint。**PartyMimic 的动作变化惩罚与历史动作作用在含参考的a上；allmimic 推荐配置作用在actor原始输出上。** 权重都为−0.1也不表示相同正则化。来源：[Party policy](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/networks.py:339)、[allmimic action](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/task.py:233)。

PartyMimic 采集时在动作上加 N(0,0.1²)，再裁剪至[−20,20]；想象路径没有这个动作裁剪。allmimic 主路径未做对应[−20,20]裁剪；推荐配置25%环境没有动作探索噪声，但这些环境仍有观测噪声和动力学随机化。

两边基础奖励形式相同：`r = dt × [Σ w_i exp(−e_i/σ_i²) − 0.1 Σ(Δa)² − 2 Σ关节超软限位量]`。位置/速度误差为body上平均的平方欧氏距离，姿态为平方旋转角；根误差为全局根位姿误差。body位置/姿态先对齐机器人xy和yaw，body速度直接比较世界坐标速度。

| 奖励项 | 权重（两边） | PartyMimic 训练σ | allmimic σ / Party标准评估σ |
|---|---:|---:|---:|
| body位置 | 1 | 0.105 m | 0.3 m |
| body姿态 | 1 | 0.14 rad | 0.4 rad |
| body线速度 | 1 | 1.0 m/s | 1.0 m/s |
| body角速度 | 1 | 3.14 rad/s | 3.14 rad/s |
| root位置 | 0.5 | 0.105 m | 0.3 m |
| root姿态 | 0.5 | 0.14 rad | 0.4 rad |
| 动作变化 | −0.1 | 平方和；动作含参考skip | 平方和；推荐配置用原始actor输出 |
| 软关节限位 | −2 | 关节范围的90% | 关节范围的90% |
| 接触匹配/防滑 | Party想象阶段0.25 | 开启 | 无 |
| 残差幅值 | Party无对应默认项 | — | 可选w_residual，默认0 |

PartyMimic 的 σ 缩放只作用于4项位置/姿态，**不是所有奖励核都乘0.35**。例如body位置误差0.1m时，单独这一项从 `exp(−0.1²/0.3²)≈0.895` 变成 `exp(−0.1²/0.105²)≈0.404`。

接触项使用脚踝原点高度的平滑sigmoid（阈值0.045m，温度0.005m），并匹配参考接触状态；参考接触由该片段脚高度5%分位数+0.015m及水平速度<0.5m/s判定。参考接触时还奖励低水平脚速，速度核0.3m/s。真实采集奖励使用窄核但不加接触项；想象奖励及SIL重算回报加入接触项，标准评估恢复宽核且不加接触项。接触项不是接触力传感器读数。

来源：[Party 奖励](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/mdp.py:139)、[训练奖励配置](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/train.py:169)、[allmimic 奖励](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/task.py:353)。

两边每步都乘dt=0.02，六个正向基础项的最大和为5，所以基础最大步奖励0.1。不能直接用双方训练日志的归一化return横向判断策略好坏：核宽、动作语义、接触项和日志统计口径都不同。两边均没有基础显式能耗/力矩/自碰撞惩罚项；是否发生自碰撞仍由物理模型决定。

**6. 终止、初始化与重置采样**

| 项目 | PartyMimic | allmimic |
|---|---|---|
| 根高度终止 | 与参考高度差绝对值>0.25m | 相同 |
| 倾斜终止 | abs(R_robot[2,2]−R_ref[2,2])>0.8 | 相同 |
| 想象软终止 | 高度温度0.02、倾斜温度0.04 | 相同 |
| 训练回合 | 最长500控制步=10s，或clip结束 | 相同 |
| 初始化 | 从参考随机时间帧开始，叠加扰动 | 相同思路 |
| reset关节裁剪 | 模型硬限位 | soft限位 |
| 想象起点 | 筛选剩余至少32帧的起点 | clip尾部使用valid mask |

倾斜阈值0.8是矩阵元素差，**不是0.8 rad的倾角**；倒立参考本身不会因为“倒立”就被这项终止。

两边均按1秒bin做重置采样，基础混合比例为50%均匀、40%低回报softmax、10%历史困难分布，softmax系数25，但细节不同：

| 重置采样细节 | PartyMimic | allmimic recommended |
|---|---|---|
| 多样性特征 | 5维：高度均值/标准差、水平速度、yaw速率、关节速度 | 工具12维，包括高度、倒立、竖直速度及手/膝/身体接触比例 |
| 聚类 | 启动时k-means，默认24类 | 外部工具默认256类；需传入bin-prior |
| 类质量 | n_c^0：非空类等概率 | 推荐n_c^0.3；裸默认指数0.5 |
| 应用位置 | 替换基础混合中的50%均匀分支 | prior乘在整个混合概率上，再归一化 |
| 失败回溯 | 默认30%质量分给近期失败点前一bin | 无对应默认分支 |
| softmax限幅 | 无对应上限 | softmax分量每bin上限50/num_bins |
| 历史困难分布 | EMA：0.999旧+0.001新 | max(0.999旧, 当前softmax) |
| 回报统计 | 折扣回报；非失败截断用平均奖励补尾；EMA旧权重0.8 | 归一化折扣回报统计，clamp至[0,1]；EMA新权重0.1 |

当失败统计非零时，PartyMimic 的基础50/40/10分支先整体乘0.7，再混入0.3失败分支。`div_tau=0` 表示类别等质量；`div_k=0` 才关闭聚类，不能混为一谈。allmimic 推荐preset不自动生成或加载bin-prior。限幅也只保证softmax分量，并非最终混合概率的严格上限。

来源：[Party 采样](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/env.py:198)、[allmimic 采样](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/sim_env.py:347)、[allmimic 聚类工具](/mnt/dataset/lqi3lr/dyh/allmimic/tools/bin_clusters.py:1)。

**7. 域随机化、噪声和延迟**

默认RP1没有额外覆写DomainRandCfg/SensingCfg，因此与PartyMimic的基础G1设置同类，不是重随机化版本。

| 项目 | PartyMimic默认G1/RP1 | allmimic G1 | allmimic EVT |
|---|---|---|---|
| 滑动摩擦 | U[0.3,1.2]，每环境启动一次 | U[0.3,1.2]，每回合 | XML名义摩擦×U[0.8,1.2]，每回合 |
| 摩擦施加位置 | 默认所有机器人geom；地面置0.3 | 脚部geom | 脚部geom及地面 |
| CoM偏移x/y/z | ±0.025/±0.05/±0.05m，启动一次 | 同范围，每回合 | 同范围，每回合 |
| CoM作用体 | G1 torso_link；RP1 waist_yaw_link | torso_link | waist_pitch_link |
| 编码器偏移 | 每关节±0.01rad，每回合 | 同幅度，每回合 | 同幅度，每回合 |
| 质量、惯量、PD、armature随机化 | 默认关闭 | 主环境无对应配置 | 主环境无对应配置 |
| 训练推力 | 默认无，当前图采集入口禁止 | 主环境无训练推力项 | 同左；push_test是单独测试工具 |
| 动作延迟 | 默认无；当前训练不能视为开箱即用支持 | opt-in，0～4物理子步 | opt-in，0～4物理子步 |
| 观测延迟/滤波 | 默认无；训练入口断言禁止stateful sensing | 无对应默认项 | 无对应默认项 |

摩擦表描述写入模型的参数；MuJoCo 实际接触参数还取决于成对geom的priority/混合规则，不能把某个被随机化geom的μ直接当成全部接触的有效μ。

两边重置扰动半幅基本相同：位置 `(0.05,0.05,0.01)m`；旋转 `(0.1,0.1,0.2)`；线速度 `(0.5,0.5,0.2)m/s`；角速度 `(0.52,0.52,0.78)rad/s`；关节角 `0.1rad`；不额外扰动关节速度。旋转实现不同：Party用旋转向量SO(3)指数，allmimic用XYZ欧拉转四元数再左乘，采样分布并非严格相同。

| 观测独立均匀噪声半幅 | 两边基础值 |
|---|---:|
| 关节位置 | 0.01rad |
| 关节速度 | 0.5rad/s |
| 陀螺仪角速度 | 0.2rad/s |
| 投影重力分量 | 0.05 |
| 上一步动作 | 0 |
| 参考姿态6D分量 | 0.05；allmimic仅在该特征启用时 |

PartyMimic 的底层确实定义质量/惯量、PD增益、接触solref、armature、damping、IMU、滤波、观测延迟和推力扩展；但主训练入口有 `assert not pipe.stateful`、`assert not env.push_enabled`，masked reset也限制扩展随机化项，并声明不支持执行器延迟。因此不能依据dataclass字段就声称当前配方已支持全部扩展。

allmimic `root-feedback-delay` 是实际接通的路径：腿/脚/腰/臂4组在回合开始各自采样0～4个5ms子步，即0～20ms；前若干子步保持旧target，然后切换新target。WM/critic的xi增加4维delay与29维旧setpoint，33→66。评估默认 `delay_eval=max`，所以即便关闭其他随机化，延迟仍可能保留20ms；设 `none` 才关闭这一评估延迟。

来源：[Party DR](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/dr.py:27)、[Party sensing](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/sensing.py:62)、[训练限制](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/train.py:174)、[masked reset限制](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/env.py:257)、[allmimic DR](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/sim_env.py:316)、[动作延迟](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/action_delay.py:27)。

**8. 训练超参数和优化目标**

| 参数 | PartyMimic默认 | allmimic裸默认 | allmimic推荐 |
|---|---:|---:|---:|
| num_envs | 4096 | 4096/卡 | 同左 |
| replay容量 | 1000×4096条transition | 同左/卡 | 同左 |
| warmup采集步 | 64 | 64 | 64 |
| UTD | 0.4 | 0.4 | 0.4 |
| imagination horizon | 32步=0.64s | 相同 | 相同 |
| history / goal帧数 | 10 / 10 | 10 / 10 | 10 / 10 |
| policy batch | 2048 | 4096/卡 | 4096/卡；README双卡示例显式改2048 |
| WM batch / batches | 2048 / 5 | 2048 / 10 | 同左 |
| critic epochs / minibatches | 5 / 4 | 5 / 4 | 5 / 4 |
| gamma / lambda | 0.99 / 0.95 | 相同 | 相同 |
| 基础学习率 | 4e−4 | 2e−4 | 2e−4 |
| policy初始学习率 | 1.6e−3（4倍） | 2e−4 | 2e−4 |
| cosine衰减终点 | 250,000自身iteration | 250,000自身iteration | 同左 |
| 最终LR比例 | 0.03 | 0.03 | 0.03 |
| policy loss主项 | −mean(J) | −mean(J)/32 | −mean(J)/32 |
| policy梯度norm clip | 1.0 | 1.0 | 0.03125=1/32 |
| policy梯度逐元素clip | ±0.1，可关 | ±0.1 | ±0.1 |
| WM / critic norm clip | 1.0 / 1.0 | 相同 | 相同 |
| policy EMA | 0.995 | 无 | 无 |
| SIL | 1.0；优势权重上限5 | 无 | 无 |
| value-gradient consistency | 0.5；首epoch，每minibatch最多4096状态 | 无 | 无 |
| value clip | 无对应启用项 | 关闭 | ±10，作用于bootstrap/target价值 |
| 默认iterations | 12,000 | 320,000 | 320,000 |
| checkpoint间隔 | 2000 | 4000 | 4000 |
| 定期训练内评估 | 主入口没有对应eval_every | 每4000轮 | 每4000轮 |
| MLP混合精度 | BF16，积分/奖励FP32 | 相同 | 相同 |

PartyMimic策略LR倍率在训练return达到0.75或最晚1400轮时触发，用300轮从4降至1；基础LR仍走cosine。allmimic多卡默认把LR乘sqrt(world_size)。

优化器细节：两边policy的Adam部分β=(0.9,0.95)，WM/critic的Adam部分β=(0.9,0.999)，默认无weight decay。Party的policy默认连首尾Linear权重也用Muon（pi_muon_io=True），WM只在中间Linear权重上用Muon，其余参数用Adam；allmimic可选Muon时两种网络都只覆盖中间Linear权重。两边Muon均用match_rms_adamw匹配更新尺度，不能仅设置同名optimizer就认为参数分组相同。

两边的J均为带软存活概率的折扣奖励和，加末端critic bootstrap。Party的SIL用真实记录回报优于V的样本回归历史动作，按动作缩放换算成关节角误差；不是对所有参考动作做监督BC。其价值梯度一致性使critic状态梯度接近想象回报的BPTT梯度。allmimic推荐value clip用于限制bootstrap/λ-return中的价值，不是把critic输出网络改成tanh；其critic回归还按想象存活概率和有效帧加权，Party默认是MSE再加梯度一致性项。

推荐配置的allmimic虽然norm clip是Party的1/32，但loss本身也除以32；只比较clip数字会误判约束强度。两边优化器及额外损失不同，不能据此断言更新完全等价。

**iteration 的计数必须换算。**

| 同一学习预算的计数 | PartyMimic | allmimic |
|---|---|---|
| 一轮iteration | 平均采集2.5步，再做1次学习阶段 | 采集1步；累计UTD信用后做学习 |
| 12,000次学习阶段 | 12,000 iterations | 约30,000 iterations |
| 默认总学习阶段数 | 12,000 | 约128,000 |
| 默认向量采集步，含64步warmup | 30,064 | 320,064 |
| 默认单卡真实transition | 123,142,144 | 1,310,982,144 |

所以相同 `lr_decay_iters=250000` 在两边也对应不同的采样/更新预算，不能直接复制。不同迭代计数下的it/s和checkpoint编号同样不能直接比较。来源：[Party主循环](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/train.py:1082)、[allmimic主循环](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/train.py:766)。

完整CLI默认值与allmimic各预设覆盖值另存于 [comparison_training_parameters.csv](/mnt/dataset/lqi3lr/dyh/PartyMimic/docs/comparison_training_parameters.csv)，由AST提取，不执行训练模块。

**9. 仿真模型、控制频率与RP1/EVT电机参数**

| 项目 | PartyMimic | allmimic |
|---|---|---|
| 物理/控制周期 | 5ms / 20ms，4子步 | G1/EVT同为5ms / 20ms，4子步 |
| G1选择 | g1mj用mjlab模型；g1用本地URDF转换模型 | mjlab G1 |
| 自碰撞 | g1mj、RP1开启；旧g1配置显式禁用 | 当前安装mjlab G1为FULL_COLLISION；EVT按XML碰撞配置 |
| 控制 | BuiltinPositionActuator，位置PD | G1位置PD；EVT每5ms显式计算力矩PD |
| solver配置 | implicitfast / pyramidal / newton，iterations100，ls50，tol1e−6，ccd35 | G1显式iterations10、ls20；EVT仅显式dt，其余框架默认 |
| nconmax / njmax | 默认128 / 500 | G1 35 / 250；EVT 128 / 1024 |
| 依赖声明 | mjlab≥1.6、torch固定2.9.0、warp1.16.0 | mjlab1.2.0、torch2.14.1、MuJoCo3.5.0、warp1.12.0 |

这里“100次/10次solver迭代”是配置上限，不代表每个物理步都执行满次数。PartyMimic当前venv缺少torch，未启动其完整仿真；allmimic已安装环境读取为torch2.14.1+cu130、MuJoCo3.5.0。对g1mj实际依赖模型的一致性，不能只凭同为G1断定；应固定mjlab版本和导出的MJCF。

RP1执行器参数来自Python配置，会覆盖相应XML执行器参数：

| RP1关节组 | Kp | Kd | 力矩上限Nm | armature | α(rad/action) |
|---|---:|---:|---:|---:|---:|
| 腰roll/yaw、髋pitch/roll、膝 | 120 | 7.5 | 140 | 0.0290856 | 0.291667 |
| 髋yaw、肩pitch | 35 | 2.5 | 55 | 0.00884 | 0.392857 |
| 踝pitch/roll | 25 | 1.5 | 64 | 0.006266268 | 0.64 |
| 肩roll/yaw、肘、腕roll/yaw/pitch | 12.5 | 0.75 | 32 | 0.00313313 | 0.64 |

RP1统一joint frictionloss=0.1，XML基础joint damping=0.01；α=0.25·effort/Kp。28DoF来自腿12+腰2+臂14。默认站姿中左右髋pitch为−0.1/+0.1、膝−0.3/+0.3、踝pitch均−0.2、肩pitch+0.2/−0.2、肩roll+0.2/−0.2、肘−1.2/+1.2，未指定项为0。左右关节轴不能套G1的同号规则，镜像符号由RP1模型轴向推导。

EVT使用config.json的Kp/Kd，但有效力矩限制取编译模型的motor ctrlrange、forcerange、joint actfrcrange交集。下面是本次CPU编译后读取的值，左右对称，α全部0.25：

| EVT关节组 | Kp | Kd | **有效**力矩上限Nm | armature |
|---|---:|---:|---:|---:|
| 髋pitch/roll | 300 | 15 | 200 | 0.24 |
| 髋yaw | 150 | 7.5 | 142 | 0.18 |
| 膝pitch | 350 | 15 | 330 | 0.37 |
| 踝pitch | 30 | 3.75 | 55 | 0.032 |
| 踝roll | 16.8 | 2.1 | 50 | 0.032 |
| 腰yaw | 400 | 7.5 | 91 | 0.17 |
| 腰roll/pitch | 400 | 15 | 91 | 0.17 |
| 肩pitch | 150 | 5 | 90 | 0.1 |
| 肩roll | 50 | 2.5 | 60 | 0.1 |
| 肩yaw | 50 | 2.5 | 36 | 0.1 |
| 肘pitch | 150 | 5 | 50 | 0.1 |
| 肘yaw | 50 | 5 | 25 | 0.1 |
| 腕pitch/roll | 20 | 2 | 25 | 0.0236 |

EVT passive damping=1、frictionloss=0.1。29DoF为腿12+腰3+臂14，但其臂结构有elbow_yaw、只有两个wrist关节，和G1的关节拓扑也不同。默认站姿髋pitch−0.25、膝0.5、踝pitch−0.25、肩roll左+0.3/右−0.3、肘pitch−0.3，其余0。

例如config.json写髋pitch effort=235，实际编译限幅为200；腕写36，实际为25。只对比JSON或只读motor ctrlrange都会漏掉另一层限制。来源：[EVT motor_parameters](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/robots/evt.py:99)、[EVT力矩PD执行](/mnt/dataset/lqi3lr/dyh/allmimic/allmimic/sim_env.py:468)。

G1两边采用同类自然频率10Hz、阻尼比2的BeyondMimic执行器设计，Kp=armature·(2π10)²，Kd=4·armature·2π10，α=0.25·effort/Kp；Party本地G1各电机组的armature/effort见 [g1_constants.py](/mnt/dataset/lqi3lr/dyh/PartyMimic/src/partymimic/g1/g1_constants.py:25)。g1mj实际执行器仍来自其安装的mjlab。

**10. 数据、部署、工程和评估**

| 方面 | PartyMimic | allmimic |
|---|---|---|
| 主运动输入 | 一个目录下多clip；npz根位姿+关节，或G1 CSV | 聚合npz运动库 |
| 运动采样 | 加载时重采样50Hz，计算速度及跟踪body FK | 读取预处理速度；验证dt=0.02及机器人/关节顺序 |
| 数据内存 | 预存跟踪body位置、旋转、速度，FP32 | 支持compact_v1；根位置FP32，其余主体FP16，frame返回FP32 |
| README训练配方 | LAFAN1 + BONES多样性10k + Kungfu，约41h数据 | LAFAN1 / hard306 / 完整BONES训练集等多个配方 |
| 数据筛选 | 多样性选片、地面/穿透/速度QA、R2R | infeasible whitelist、held-out split、bin prior |
| 训练检查点 | pi可为EMA，另外存pi_live；last含优化器 | policy为当前策略；保存解析后的cfg、优化器及机器人/物理元数据 |
| 微调 | --init，重置优化器和统计；--resume继续 | --resume继续；另有adapt.py |
| 多GPU | 两阶段流水，policy用上一轮WM/replay | torchrun数据并行；梯度/归一化统计同步 |
| 编译 | 采集、WM、policy/critic更新图捕获；MLP goal提前批量编码 | 推荐编译单想象步和WM loss；可选BPTT checkpoint节省显存 |
| 评估/导出 | sim2real接口ONNX+YAML、sim2sim工具 | 原生评估、ONNX、EVT CPU MuJoCo sim2sim和动力学诊断 |
| 数值防护 | policy非有限更新门控 | 额外仿真状态NaN清理、坏transition排除、WM/critic梯度检查 |
| 测试 | 仓库未见同类tests目录 | tests下有CPU/GPU、延迟、根反馈、导出、动力学测试 |

不要直接把PartyMimic README中的“26分钟、held-out body error毫米”与allmimic的“6.6小时、306困难动作零失败”相除：训练数据规模、机器人/模型、迭代单位、GPU数量、终止/评估协议和指标均不同。仓库宣称的结果本次没有重新训练验证。

若要比较算法效果，建议先固定同一个机器人MJCF和控制器、同一训练/测试clip与重采样方式、同样50Hz/5ms物理步、随机种子和GPU预算；同时记录真实transition、policy更新次数、WM优化步数、想象样本数与实际时间。指标同时报body位置误差、root漂移、完成率、足滑和关节/力矩超限比例，再逐个消融窄核、结构化WM、Muon、EMA、根反馈与动作残差范围。

迁移allmimic配置时，优先对齐观测排列、goal排列、动作单位和PD模型，然后考虑Party配方。对EVT不建议直接套用RP1的PD/armature或全身参考skip；推荐配置保留下肢绝对动作是有意设计，需作为独立消融项检验。

**11. 核验范围**

本次完成源码调用链审阅、CLI/预设默认值AST提取、网络CPU参数量核验、RP1 XML关节数核验、allmimic G1/EVT模型CPU编译及EVT有效控制参数读取。未启动GPU训练或完整rollout，未改动两边训练实现。Party当前venv缺torch，因此其网络核验使用allmimic的PyTorch环境独立执行抽取的网络类，不能视为Party依赖环境完整可用的证明。

没有指定某个run或checkpoint，所以本报告不把历史runs中的覆盖参数当成本次默认配置；实际复现实验还应以该run保存的args/cfg为准。
