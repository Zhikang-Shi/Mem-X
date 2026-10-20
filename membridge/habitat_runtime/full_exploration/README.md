# Habitat 前沿探索：方向试探、无动作恢复与旧图兜底

此目录是 `memory_module` 在 Habitat 中实际执行前沿探索与方向试探的入口，
由 `habitat_runtime/trainer.py` 安装；不依赖外部 SG 源码或实验目录。

## 默认配置

从 2026-09-06 起，已开启前沿探索和方向试探的运行使用以下组合：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MIN_PROGRESS_M` | `-0.26` | 允许单步暂时远离目标，保留安全检查 |
| `LIFELONG_SG_EXPLORE_NO_ACTION_RECOVERY` | `1` | 配置后的普通试探仍无动作时启用恢复 |
| `LIFELONG_SG_EXPLORE_OLD_MAP_FALLBACK` | `1` | 普通试探及原目标复查失败后，尝试旧图其他区域 |
| `LIFELONG_SG_EXPLORE_OLD_MAP_MAX_ACTIONS` | `48` | 第二层兜底追加的总动作上限 |
| `LIFELONG_SG_EXPLORE_OLD_MAP_MAX_FORWARDS` | `36` | 第二层兜底追加的前进上限（包含在总动作内） |
| `LIFELONG_SG_EXPLORE_OLD_MAP_SWITCH_ACTIONS` | `8` | 检查旧图方向进展的动作间隔 |
| `LIFELONG_SG_EXPLORE_OLD_MAP_MIN_PROGRESS_M` | `0.25` | 间隔内向接入点接近不足此值则换区域 |
| `LIFELONG_SG_WORLD_FULL_WINDOW_RETRY` | `1` | 世界地图路径规划失败后扩大窗口重试一次 |

这些默认值不自动开启整个探索模块。`LIFELONG_SG_EXPLORATION_ENABLE`
和 `LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE` 的总开关、全局动作/距离预算不变。
窗口重试属于几何路径规划，独立于探索总开关，缺地图/无历史轨迹时不扩大。
阈值只影响方向试探的目标进展筛选，不改变前沿评分、障碍净空或物理动作检查。

显式环境变量优先于默认值。旧实验 `formal.env` 中若指定 `-0.03`，仍按该值运行。
模块提供 `probe_defaults.env`；Docker 启动时将它作为**最后一个** `--env-file`
传入即可覆盖旧值，不需要修改历史实验配置。宿主 shell 的环境不会自动传入 Docker。

复现旧机制时，在容器环境显式设置：

```bash
LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MIN_PROGRESS_M=-0.03
LIFELONG_SG_EXPLORE_NO_ACTION_RECOVERY=0
LIFELONG_SG_EXPLORE_OLD_MAP_FALLBACK=0
LIFELONG_SG_WORLD_FULL_WINDOW_RETRY=0
```

仅测试单项时可独立设置两项。此前 21 组离线实验比较的是“仅放宽”和
“旧阈值 + 恢复”两个对照，不能直接把其指标当作本联合默认配置的效果。

## 执行流程与边界

1. 优先执行普通前沿/试探逻辑；有动作时不进入恢复。
2. 仅 `no_certified_progressing_single_step` 触发 `probe_recovery.py`，
   预算耗尽、缺地图、缺 navmesh 等状态不会借此绕过检查。
3. 使用最近真实深度帧补充近处自由空间证据，并认证已实际占据的身体足迹。
   缓存最多 12 帧、跨 episode 清空；普通观测阶段只缓存，不额外修改地图。
   未知/不确定格需要至少两帧自由证据，身体足迹外的已知障碍不被覆盖。
4. 在已观测自由且满足净空的单步路径中选择局部绕行，仍使用 Habitat
   `try_step` 验证实际单步位移，不从完整 navmesh 获取到目标的全局路线。
5. 恢复短段每前进 4 次重新尝试普通筛选；能恢复即交回原流程。
   每 4 个试探动作复查连通，无动作退出前再复查一次，避免只补图已连通却退出。
6. 仍找不到安全动作时最多补一圈 12 次观察转向；普通恢复的转向和前进
   共同消耗已有试探预算，不重置计数。普通恢复停止后才考虑下一层旧图兜底。

## 第二层：旧图区域兜底

`old_map_fallback.py` 只在普通试探无动作/预算耗尽，且原目标最后复查仍失败、
当前地图也未获旧图连接证明时启动。缺地图、缺 live navmesh 等失败不能触发。
每个目标状态最多启动一次，不扩大 episode、探索总动作或总距离预算。

- `online_adapter.start_episode` 在预加载完成、当前 episode 首帧之前冻结旧轨迹。
  EP2 的冻结集合包含已有 EP0/EP1；在线新增轨迹不会被误认作旧图。
  外部自建 adapter 需显式调用 `navigation.map_connection.freeze_episode_history`；
  未标记 episode 边界时不启用旧图兜底，而不是把当前图猜作旧图。
- 原目标 UID、物体中心及历史观察点保持不变，仅改变试探方向。
  优先近处同层旧轨迹区域，8 个动作内接近不足 0.25 m 则换空间/角度不同的区域。
- 复用原有安全试探和真实深度恢复，每次尝试仍检查已知自由中心线、障碍净空和
  Habitat 单步位移；不使用未观测的全局 navmesh 路径。
- 第二层独立追加最多 48 动作 / 36 前进，限额仅存在于当前目标状态中，不污染
  evaluator 全局配置。换方向保留访问和碰撞证据；不能重置追加预算。
- 每 4 个试探动作及退出前先检查原目标。目标可规划时提交该目标的路径/窗口，
  清除逃逸状态并恢复原 SG 控制器；不会输出“旧点到达=任务成功”。
- 仅旧图可规划而原目标仍失败时，结束旧图兜底并交还原失败/LaViRA 回退处理。
  全部冻结同层旧节点参与接图检查，结果须有原 exact planner 的路径证据。

## 路径规划失败后扩大窗口

`trainer._sg_plan_world_path` 先完整执行原 `_sg_plan_world_path_once`。
原路径成功直接返回；仅几何连通失败时，将已观测同层轨迹的范围加入窗口锚点，
再调用同一规划器一次。没有更大范围则不重复执行。

重试不改目标、exact/snap 参数、安全距离、障碍层或候选筛选，仍受
`LIFELONG_SG_WORLD_MAX_WINDOW_CELLS` 上限约束。此入口同时覆盖初始目标规划、
探索后的目标复查和执行过程中的重规划，避免接图检查与目标规划窗口不一致。
它可能提前修复原本会进入探索的失败路径，这是窗口修复的预期效果。
结果中的 `world_window_retry` 记录初次失败、扩大范围的锚点数量与重试状态。

此前 17/21 是离线旧图兜底的规划连通结果，不能直接当作本次正式接入后的
端到端 SR；也不代表 17 组全部依靠新兜底或已经实际走到旧轨迹。

没有新增 SG 检索、VLM 调用或语义模型加载。安全检查并不保证实际零碰撞；
碰撞反馈和实际位移仍由运行环境记录。只有进入恢复后才补图并改变局部动作。

## 验证

在 `LifeLongHomeNav` 目录运行，不需要模型、API 或 GPU：

```bash
python -m unittest memory_module.habitat_runtime.full_exploration.test_probe_recovery -v
python -m unittest memory_module.habitat_runtime.test_full_exploration_patch -v
python -m unittest memory_module.habitat_runtime.full_exploration.test_old_map_fallback -v
python -m unittest memory_module.navigation.test_map_connection -v
```

测试覆盖默认值、环境覆盖、普通动作优先、禁用等价性、恢复回交、预算上限、
未知/障碍拒绝、补图证据、退出连通复查，以及联合配置的几何试探。

### 2026-09-06 集成验证

- 宿主完整模块测试：103 项通过（另有 19 个 subtests）。
- 实际 `lavira-sg` / Python 3.8 容器：36 项相关回归测试通过。
- `memory-integrated-oldmap-smoke-20260906-02`：3 个真实案例分别用保存深度和
  重新渲染深度恢复，共 6 次 smoke，动作/位姿/结果配对一致。
  其中一例的规划器展开数和可达栅格数各差 1；目标落点、路径长度及交接结果
  一致，不要求两种深度量化方式的内部栅格计数完全相同。
  - `00815_level3r_000__query_ep0`、`00843_level3r_002__query_ep2`：原窗口失败，
    扩窗重试成功，无需额外移动，原目标路径成功交接。
  - `00848_level3r_001__query_ep1`：旧图兜底追加 44 动作、12 次前进，实际约
    2.54 m，恢复原目标路径并清除逃逸状态。
- smoke 使用真实几何规划、探索控制器和路径提交；交接后的导航动作回调替换为
  不执行的测试标记，没有运行完整 LaViRA 后半段，不是全量 SR 评测。
- 输出在 `/share/shared/shizk/LifeLongHomeNav/level3/outputs/` 的同名目录，本地
  `level3/outputs/` 保留软链接。运行代码哈希与 M1 输入哈希校验通过。
- `...-01` 保留为失败测试记录：第二种深度恢复错误地比较了第一种深度的内部
  拒绝计数；修正测试参考路径后用独立 `...-02` 重跑，模块代码未因此改变。
