# Agent 进度入口

最后更新：2026-11-21。下一个 agent 先读 [docs/AGENT_HANDOFF_2026-10-21.md](AGENT_HANDOFF_2026-10-21.md)，再读本文件第 2 节和第 6 节。不要从聊天记录重建结论。交接文档覆盖下面第 5 节的旧动作，也覆盖 `docs/next_steps_blocking_and_intervention.md` 第 2.2 节「先 MLP 再接 GRU」的顺序。甲的两轮序贯训练以第 6 节为准。乙、丙的首轮对照结果见第 7 节。

## 1. 项目是什么

仓库目录名是 DAMMFND，研究代码是 CORTGN：用时序图对突发舆情做动态阻断，用实现树上的反事实切除评估。DAMMFND（AAAI 2025，多模态假新闻检测）是引用对象，不是本仓库要复现的系统。入口说明在 `README.md`。

## 2. 已经完成的实验

书面报告：`docs/tgn_blocking_report.md`（2026-09-19）。

- 数据 `weibo22/`：4664 棵树（非谣言 2351 / 谣言 2313），约 381 万节点。与 `data/` 里的 DAMMFND 多模态表没有实例级对齐。
- 脚本：`scripts/blocking_baseline.py`、`scripts/tgn_blocking.py`。结果 json 在 `results/blocking_baseline_results.json`、`results/tgn_blocking_results.json`。
- 协议：观察窗口之后选 \(k\) 个非根节点，切掉其未来子树。主指标 `block_rate`。种子 3074。
- 全量谣言、\(k=5\)：1h 出度 10.6% / oracle 15.5%；6h 出度 13.3% / oracle 22.9%。
- 测试集谣言：1h \(k=5\) MiniTGN 11.33% / 出度 11.19% / oracle 16.01%；6h MiniTGN 13.75% / 出度 14.37% / oracle 24.75%。MiniTGN 没有形成可用增益。不要续训 `results/tgn_blocking.pt`。
- 甲的两轮 REINFORCE 见第 6 节。当前最好是「窗口结束补满名额」：测试谣言 11.50% / 出度 11.19% / oracle 16.01%。超过出度 0.31 个百分点，未到 12.2% 及格线。不要续训 `results/tgn_rl_blocking.pt` 或 `results/tgn_rl_fillk.pt`。
- 文本检测轮次：`src/text_training_round_metrics.md`，AUC 约 0.53–0.56。不能当作辟谣门控。

## 3. 2026-10-19 本次完成

导师要求：把传播阻断的准确率做高，可以试强化学习；阅读两篇论文（路径：VLDB 2026 影响力与阻断最小化；内容：AAAI 2026 证据型多智能体辩论）。

已下载 PDF 并通读，笔记和规划已落盘。没有改训练代码，没有重跑实验。

| 文件 | 作用 |
| --- | --- |
| `papers/VLDBJ2026_influence_blocker_minimization.pdf` | The VLDB Journal 35:53，2026-09-03，DOI `10.1007/s00778-026-01003-4`，26 页 |
| `papers/VLDBJ2026_influence_blocker_minimization.txt` | 上述 PDF 的逐页文本 |
| `papers/AAAI2026_ED2D_evidence_multiagent_debate.pdf` | AAAI-26 会议版，页 38542–38550，DOI `10.1609/aaai.v40i45.41196` |
| `papers/AAAI2026_ED2D_evidence_multiagent_debate.txt` | 上述 PDF 的逐页文本 |
| `papers/arxiv_2511.07267_ED2D.pdf` | 同一 ED2D 工作的 arXiv 预印本，对照用 |
| `docs/paper_notes/VLDBJ2026_influence_blocker_minimization.md` | 路径论文笔记 |
| `docs/paper_notes/AAAI2026_ED2D_evidence_multiagent_debate.md` | 内容论文笔记 |
| `docs/next_steps_blocking_and_intervention.md` | 可执行的下一步 |

论文身份不要弄混：VLDB Journal 2026 这篇是期刊扩展，包含 IMIN 的 SandIMIN 和 BM 的 Heur-BM / App-BM / App-BM+。2014 年之后的会议短文 *Efficient Influence Minimization via Node Blocking*（PVLDB 2024, arXiv:2405.12871）只覆盖 IMIN，不是导师指定的这一篇。

## 4. 读完应带走的判断

- 微博转发是树。VLDB 论文的非次模性来自多条路径的组合效应；树上 \(D_L=D=D_U\)，不要在单棵树上重写 SandIMIN。
- 现有协议只做了固定预算 IMIN。论文证明近似比不能从 IMIN 传到 BM。下一步先补「达到 \(\eta\) 的最少阻断人数」。
- 强化学习用在观察窗口内的序贯选点，奖励是 `block_rate`。论文本身没有 RL。一次性 top-\(k\) 会重复选择同一条链上的祖先和后代。
- 成功线写死在规划第 0 节：测试集谣言、1h、\(k=5\)，至少比观察出度高 1 个绝对点，且 6h 不低于出度；超过 1h oracle 16% 视为泄漏。
- ED2D 在 Weibo21 上 Acc 83.59。判错的辟谣会提高相信和转发，并能部分抵消正确的人工解释。当前文本 AUC 不够，不能直接对 weibo22 生成辩论。
- 路径 `block_rate` 和内容说服分表报告。

## 5. 下一个 agent 的默认动作

以 `docs/AGENT_HANDOFF_2026-10-21.md` 第 3 节为准，不要写论文或 PPT：

1. 戊先做共用评价函数，并实现 `scripts/blocking_bm.py`（1h/6h，\(\eta=0.10/0.15/0.20\)）。
2. 甲：TGN 记忆 + REINFORCE 序贯阻断。乙：同一套 REINFORCE，只有手算特征。丙：TGN 记忆 + 排序损失，窗口结束一次性 top-5，不续训旧权重。
3. 丁等甲的配置接口，再跑三组消融和 6h 迁移。

未收到导师要求前，不要做用户级大图上的 SandIMIN 复现，不要跑 ED2D 的 LLM 辩论。

2026-11-05 对甲的补充：第 6 节的两轮已经跑完。下一版如果继续做甲，只把出度特征改成 1 小时结束时的出度，保留窗口结束补满名额，用新的 `--tag` 另存，不要和别的改动混在同一次训练里。乙、丙仍分开做。丁和 6 小时窗口等主结果到及格线后再做。

## 6. 甲：TGN 记忆 + REINFORCE，2026-10-22 至 2026-10-29

脚本是 `scripts/tgn_rl_blocking.py`。一棵转发树是一局。每来一条观察窗口内的转发，就用 GRU 更新边两端的记忆，再给还能选的非根节点打分。训练时抽样，验证和测试时取最高分。根帖不能拦。已经拦过的人，其观察窗口内的下游不再参与选择。奖励是最终 `block_rate`；训练时有随轮次衰减的奖励塑形，验证和测试不用。早停看验证集谣言阻断比例，耐心 3，最多 20 轮。超过 1536 步的树分段反传。权重 `*.pt` 和日志 `*.log` 不进 Git。

通俗说明：

- 第一轮，每一步都可以跳过：[docs/tgn_rl_blocking_summary_zh.md](tgn_rl_blocking_summary_zh.md)
- 第二轮，窗口结束必须补满剩余名额：[docs/tgn_rl_fillk_summary_zh.md](tgn_rl_fillk_summary_zh.md)

| 版本 | 最好验证轮 | 验证谣言 | 测试谣言 | 测试平均选中人数 | 相对出度 |
| --- | ---: | ---: | ---: | ---: | --- |
| 可以一直跳过 | 1 | 9.16% | 9.43% | 2.76 / 5 | 低于 11.19% |
| 窗口结束补满名额 | 5 | 10.96% | 11.50% | 4.57 / 5 | 高于 11.19%，未到 12.2% |

两轮的测试 oracle 都是 16.01%，`leakage_void` 都是 false。第二轮验证集从第 2 轮起高于验证出度 10.63%，第 5 轮最高 10.96%；第 6–8 轮掉到 10.00% 后早停。测试加载的是第 5 轮权重。

结果文件：`results/tgn_rl_blocking_results.json`、`results/tgn_rl_selection_counts.json`、`results/tgn_rl_fillk_results.json`、`results/tgn_rl_fillk_selection_counts.json`。本机权重是 `results/tgn_rl_blocking.pt`（第 1 轮）和 `results/tgn_rl_fillk.pt`（第 5 轮）。

## 7. 乙、丙首轮对照，2026-11-21

乙、丙已完成交接中各自的对照实验。沿用 `weibo22`、种子 3074 和级联级划分，两个窗口的预算都是 `k=5`。1 小时主表有 311 棵可评测试谣言树，6 小时有 285 棵。

| 路线 | 最佳验证轮 | 1 小时验证 | 1 小时测试 | 6 小时测试 |
| --- | ---: | ---: | ---: | ---: |
| 乙，手算特征 + REINFORCE，窗口末补名额 | 7 | 11.18% | 11.88% | 未测 |
| 丙，MiniTGN + 成对排序，窗口末一次选点 | 2 | 10.67% | 11.26% | 13.65% |

乙在第 10 轮早停，1 小时比观察出度高 0.69 个百分点，仍低于 12.2% 的成功线。丙固定训练 5 轮，1 小时低于 12.2%，6 小时低于同窗口观察出度 14.37%。

- 乙的训练经过和复现命令见 [乙线说明](tgn_rl_only_fillk_summary_zh.md)，逐轮记录和测试指标见 [乙线结果 JSON](../results/tgn_rl_rl_only_fillk_results.json)。
- 丙的训练经过和复现命令见 [丙线说明](tgn_rank_summary_zh.md)，逐轮记录和测试指标见 [丙线结果 JSON](../results/tgn_rank_results.json)。1 小时主表在 `test_rumor`，两个窗口的结果在 `test_rumor_by_window`。
