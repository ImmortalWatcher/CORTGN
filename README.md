# CORTGN

Counterfactual Opinion Reasoning with Temporal Graph Networks

基于时序图网络，对突发舆情传播做**动态阻断**，并用**反事实推演**评估“若当时阻断某些节点，后续级联会少走多远”。

本仓库是独立研究代码。导师提供的 [DAMMFND](https://github.com/luweihai/DAMMFND)（AAAI 2025）只作为多模态检测的基础工作；论文若用到，按文献引用，而不是把本仓库写成它的 fork。

第一轮实验报告：[docs/tgn_blocking_report.md](docs/tgn_blocking_report.md)。

## 实验方向

1. **数据与协议** 使用 `weibo22` 的时序转发树，从观察窗口内已出现的非根节点中最多选 `k` 个阻断。`block_rate` 衡量窗口结束后被覆盖的节点比例。
2. **非学习基线**：随机、最早出现、粉丝数、观察出度，以及贪心 oracle（上界）。
3. **学习方法** 甲用 TGN 记忆和强化学习逐步选点，乙沿用同一套强化学习流程并只输入手算特征，丙用 MiniTGN 学习候选排序并在窗口末一次选点。
4. **反事实**：在实现树上做 rollback，而不是再训一个生成式传播模型。
5. **多模态（后续，非当前主线）**：DAMMFND 文本/图像检测与级联树目前对不齐，不能直接拼进阻断实验。

## 当前进度

- **级联盘点完成**：4664 棵树（非谣言 2351 / 谣言 2313），约 381 万节点、286 万独立用户。与本地多模态 `data/` **没有实例级对齐**。
- **阻断基线已跑完**（不含根节点；主看谣言树）。1 小时窗口、`k=5` 时，随机约 3.8%，观察出度约 10.6%，oracle 约 15.5%。出度已明显好于随机，但离 oracle 仍有空档。
- **MiniTGN 回归实验已完成。** 在持出谣言树上，1 小时比观察出度高 0.14 个百分点，6 小时低于同窗口出度。详见 [第一轮实验报告](docs/tgn_blocking_report.md)。
- **weibo22 级联数据已入库**（约 4GB）。图片、DAMMFND 特征 `pkl` 和预训练权重仍留在本地。

### 甲、乙、丙首轮对照

测试集只统计可评谣言树，预算为 `k=5`。1 小时窗口有 311 棵，6 小时窗口有 285 棵。

| 方法 | 1 小时阻断率 | 6 小时阻断率 |
| --- | ---: | ---: |
| 观察出度 | 11.19% | 14.37% |
| 旧 MiniTGN 回归 | 11.33% | 13.75% |
| 甲：TGN + REINFORCE，窗口末补名额 | 11.50% | 未测 |
| 乙：手算特征 + REINFORCE，窗口末补名额 | 11.88% | 未测 |
| 丙：MiniTGN + 成对排序，窗口末一次选点 | 11.26% | 13.65% |
| 事后 oracle | 16.01% | 24.75% |

三条线的首轮对照均已完成。成功线要求 1 小时达到 12.2%，同时 6 小时不低于出度基线。目前三条线均未达到成功线。

- [甲线补名额版说明](docs/tgn_rl_fillk_summary_zh.md)
- [乙线实验说明与复现命令](docs/tgn_rl_only_fillk_summary_zh.md)
- [丙线实验说明与复现命令](docs/tgn_rank_summary_zh.md)

实验脚本位于 `scripts/`，结果 JSON、CSV、人数统计和数据盘点保存在 `results/`。重新运行脚本也会写入该目录。训练日志和模型权重保留在本机，继续由 `.gitignore` 忽略。

## 接下来

导师要求把阻断做准，并可以试强化学习。路径侧看 VLDB Journal 2026 的影响力/阻断最小化，内容侧看 AAAI 2026 的 ED2D。PDF、笔记、成功线和给下一个 agent 的入口：

- [docs/AGENT_PROGRESS.md](docs/AGENT_PROGRESS.md)
- [docs/next_steps_blocking_and_intervention.md](docs/next_steps_blocking_and_intervention.md)

共用评估结果汇总和最少阻断人数（BM）表仍待完成。甲线的 6 小时迁移与消融按进度文档安排，多模态与级联对齐继续暂缓。分工见 [2026-10-21 交接文档](docs/AGENT_HANDOFF_2026-10-21.md)。
