# Agent 进度入口

最后更新：2026-10-19。下一个 agent 先读本文件，再按第 4 节打开对应笔记。不要从聊天记录重建结论。

## 1. 项目是什么

仓库目录名是 DAMMFND，研究代码是 CORTGN：用时序图对突发舆情做动态阻断，用实现树上的反事实切除评估。DAMMFND（AAAI 2025，多模态假新闻检测）是引用对象，不是本仓库要复现的系统。入口说明在 `README.md`。

## 2. 已经完成的实验

书面报告：`docs/tgn_blocking_report.md`（2026-09-19）。

- 数据 `weibo22/`：4664 棵树（非谣言 2351 / 谣言 2313），约 381 万节点。与 `data/` 里的 DAMMFND 多模态表没有实例级对齐。
- 脚本：`scripts/blocking_baseline.py`、`scripts/tgn_blocking.py`。结果 json 在 `scripts/blocking_baseline_results.json`、`scripts/tgn_blocking_results.json`。
- 协议：观察窗口之后选 \(k\) 个非根节点，切掉其未来子树。主指标 `block_rate`。种子 3074。
- 全量谣言、\(k=5\)：1h 出度 10.6% / oracle 15.5%；6h 出度 13.3% / oracle 22.9%。
- 测试集谣言：1h \(k=5\) MiniTGN 11.33% / 出度 11.19% / oracle 16.01%；6h MiniTGN 13.75% / 出度 14.37% / oracle 24.75%。MiniTGN 没有形成可用增益。不要续训 `scripts/tgn_blocking.pt`。
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

按 `docs/next_steps_blocking_and_intervention.md` 第 5 节：

1. 实现 `scripts/blocking_bm.py`（1h/6h，\(\eta=0.10/0.15/0.20\)）。
2. 实现 `scripts/rl_blocking.py`（REINFORCE，对照出度、排序回归、oracle）。
3. 内容侧先跑 `scripts/debunk_harm_threshold.py`，再决定是否对 60 条根帖调用 LLM。

未收到导师要求前，不要做用户级大图上的 SandIMIN 复现。
