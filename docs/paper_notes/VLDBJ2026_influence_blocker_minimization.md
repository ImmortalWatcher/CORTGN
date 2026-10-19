# 学习笔记：Efficient and cost-effective influence and blocker minimization

- 文献：Wang, Wu, Wang, Zhang, Qin, Zhang, Lin. *The VLDB Journal* 35:53, 2026.
- DOI：https://doi.org/10.1007/s00778-026-01003-4
- 发表：2026-09-03，开放获取。收稿 2025-09-01，修回 2026-06-21，接收 2026-08-02。
- 本地 PDF：`papers/VLDBJ2026_influence_blocker_minimization.pdf`（26 页）
- 本地抽文本：`papers/VLDBJ2026_influence_blocker_minimization.txt`
- 代码与附录：https://github.com/jhwang0116/IMIN_BM
- 关系：本文是作者 PVLDB 工作 *Efficient Influence Minimization via Node Blocking* 的期刊扩展。扩展的主体是 blocker minimization（BM）及其 bicriteria 近似。影响力最小化（IMIN）的夹心近似沿用并写完整。

阅读日期：2026-10-19。下面按「问题—算法—实验—和本仓库的接口」整理，方便直接拿去设计实验。

## 1. 要解决的实际问题

在线社交网络上，误信息从种子集合 \(S\) 往外扩散。干预手段是删掉一批节点（blocker）：指向该节点的边传播概率置 0，种子本身不能删。

论文同时做两个对偶问题，目标函数都是

\[
D(B)=\mathbb{E}[I_G(S)]-\mathbb{E}[I_{G[V\setminus B]}(S)]
\]

即阻断 \(B\) 之后，种子少影响到的期望人数。

| 问题 | 输入 | 输出 |
| --- | --- | --- |
| IMIN | 图、种子 \(S\)、预算 \(k\) | \(\lvert B\rvert\le k\)，最大化 \(D(B)\) |
| BM | 图、种子 \(S\)、阈值 \(\eta\) | 最小的 \(\lvert B\rvert\)，满足 \(D(B)\ge\eta\) |

\(\eta\) 的合法范围是 \((0,\mathbb{E}[I_G(S)]-\lvert S\rvert)\)。BM 的动机是扩散存在引爆点：只要少保护到某个绝对人数，传播就会失控，所以关心「用多少个阻断名额才能压到阈值以下」。

扩散模型默认独立级联（IC）。边概率实验里取 \(p(u,v)=1/\mathrm{in\text{-}degree}(v)\)。线性阈值（LT）下 \(D(\cdot)\) 本身次模，可以去掉夹心界、直接优化 \(D\)，采样改成 LT 的 live-edge 即可，框架不变。作者把 time-aware 扩散列为未来工作。

## 2. 难度和两个问题为什么要分开做

- IMIN 在 IC 下 NP-hard 且 APX-hard（Xie 等，2023/引用 [60]）。
- \(D(\cdot)\) 单调、**非次模**。非次模来自「组合效应」：一个下游节点要同时堵上两条路才保护得住。论文给了一个子模比 \(\psi\to 0\) 的例子，因此直接对 \(D\) 做贪心没有非平凡近似比。
- 精确计算 \(D\) 是 #P-hard，因为它含影响力计算。
- BM 同样 NP-hard（Pham 等 [41]）。

精确求解时，两个问题多项式互相归约（引理 1）：对 \(k\) 或对阈值做二分即可。**近似比不能互相传递**（引理 2）。一个 \(\alpha\)-近似的 IMIN 算法可以一直达不到 \(\eta\)，基数可以任意差；一个 \(\beta\)-近似的 BM 算法拿去解固定预算 IMIN，比值可以任意接近 0。所以「预算 \(k\) 下的 block_rate」和「达到某个削减比例所需的最少阻断数」必须分开报、分开设计算法。

## 3. 已有方法及其缺口

IMIN：

- Wang 等：蒙特卡洛估计 \(D\)，再贪心。能做，但太慢。
- Xie 等：把多种子收成一个虚拟源，在随机实现上建支配树（dominator tree）。单节点 \(u\) 保护的期望人数等于它在支配树上的子树大小期望。由此得到 Advanced-Greedy（AG）和 Greedy-Replace（GR）。GR 会优先考虑种子的出邻居，效果通常好于 AG，更慢。
- 缺口：每选中一个 blocker 就要在残图上重采样、重建支配树，\(k\) 大时很贵。Youtube、Pokec 上 AG/GR 24 小时内跑不完。而且没有近似保证。

BM：

- 只有 Pham 等的 Base：在 DFS 树上估计 \(\tilde D\)，贪心加节点直到估计值 \(\ge\eta\)。样本固定约 1000。\(\tilde D\) **有偏**，阈值满足率经常低于 1（EmailCore 上甚至低于 0.8），基数也没有保证。

## 4. IMIN：SandIMIN

夹心策略（Lu 等；Wang 等）：给非次模的 \(D\) 做次模下界 \(D_L\) 和次模上界 \(D_U\)，分别求 \((1-1/e-\epsilon)\) 近似解 \(B_L,B_U\)，再加一个几乎不花钱的启发式 \(B_R\)，最后用影响力估计挑三者中残存影响力最小的那个。

多种子处理：新建虚拟源 \(s\)，向每个真种子连概率为 1 的边，使 \(\mathbb{E}[I_G(s)]=\mathbb{E}[I_G(S)]+1\)。阻断仍只能从 \(V\setminus S\) 里选。这比 Xie 等预先合并入边概率更省预计算。

下界 \(D_L\)：丢掉组合效应，只计算「单独阻断某一个节点就能保护」的节点，再对 \(B\) 取并集。单调次模。单节点上 \(D(v)=D_L(v)\)。

上界 \(D_U\)：把「所有从源到 \(v\) 的路都被堵」放松成「至少有一条路被堵」（alternative-protected）。这等价于在源可达子图上做影响力最大化，因此单调次模。

图 2 的例子：\(D_L=6\)，\(D=7\)，\(D_U=8\)。界紧的时候夹心才有用；常数上界虽然次模，但解可以是任意的。

定理 5：返回的 \(B\) 以至少 \(1-3\delta\) 的概率满足

\[
D(B)\ge \tau\,(1-1/e-\epsilon)\,D(B_1^\circ)
\]

其中 \(\tau\) 是数据相关系数，

\[
\tau=\max\Big\{\frac{D_L(B_L^\circ)}{D(B_1^\circ)},\frac{D(B_U)}{D_U(B_U)}\Big\}\cdot\frac{1-\gamma}{1+\gamma}.
\]

\(B_1^\circ\) 是 IMIN 最优解。\(\tau\) 离 1 越近，保证越接近经典次模贪心。

### 4.1 怎么估计两个界

普通 RIS 均匀抽节点，这里只关心会被种子激活的节点，而激活概率本身 #P-hard，所以不能直接套 RIS。

- **CP sequence（估 \(D_L\)）**：一次随机实现 \(\phi\) 上，对每个源可达节点 \(v\)，公共路径集是「所有 \(s\to v\) 路径的公共节点」（去掉种子和虚拟源）。这些节点就是单独阻断就能保护 \(v\) 的位置。CP-Gen 不枚举路径，而是用 Lengauer–Tarjan 建支配树，沿 DFS 把 `idom` 链并起来。无偏估计：\(D_L(B)=\mathbb{E}[\mathrm{Cov}(B)/\lvert C_s\rvert]\)。
- **LRR set（估 \(D_U\)）**：只在源可达区域内做反向可达采样。无偏估计按 \(\lvert V'_s\rvert\cdot\mathrm{Cov}/\lvert L\rvert\) 缩放。

样本数用鞅浓度自己推停止条件，不能照搬 OPIM-C，因为估计量形式不同。

- App-LB：最大化 \(D_L\)，\((1-1/e-\epsilon)\)，概率 \(\ge 1-\delta\)。
- App-UB：最大化 \(D_U\)，同样保证。期望时间 \(O\big((k\ln n+\ln(1/\delta))(m+n)/\epsilon^2\big)\)。
- Heur-IMIN：只在种子的出邻居里按 \(s(v)=\Pr[s\to v]\cdot\mathrm{deg}[v]\) 取 top-\(k\)。没有理论贡献，几乎不增加时间，但可能把 SandIMIN 的经验效果抬一下。出邻居数量不超过 \(k\) 时直接返回全部出邻居。

SandIMIN- 是关掉 App-UB 的简化版，更快，但没有可计算的近似比下界（因为那个下界要用到 \(B_U\)）。

## 5. BM：Heur-BM、App-BM、App-BM+

Heur-BM 仍是「边际增益最大的节点一直加，直到达标」，但估计量无偏，所以能证明阈值以高概率成立。两阶段：

1. 选择：用 CP sequence 估计单节点边际 \(D(v\mid B)\)。选中后只重生成与已选 blocker 相交的那些 CP（Update-CP），避免整批重采样。
2. 验证：用 RIS + stopping rule 分别估计阻断前、阻断后的影响力，用 \((1\pm\gamma)\) 界推出 \(D(B)\ge\eta\) 的保守判据。不达标就继续加节点。

代价：停机偏保守，阈值满足比经常明显大于 1，阻断人数偏多。引理 11 说明这种贪心在非次模目标上**没有常数因子的基数保证**。最坏情况是「几个节点合在一起保护一大片，但各自边际增益很小，贪心看不见」。实验里这种图很少见：Heur-BM 的人数比 App-BM+ 多出来不到一个数量级。

App-BM / App-BM+ 不去直接优化 \(D\)，而是用次模上界 \(D_U\) 做集合覆盖式的 Thre-Greedy，从而套用次模覆盖的基数界（定理 4）。

- App-BM：固定误差 \(\mu\)。\(\mu\) 必须小于一个通常很小的边界，样本才会够，否则定理 12 的保证不成立。默认 \(\mu=10^{-2}\) 在多数数据上越过了这个边界，所以 App-BM 主要是理论算法，只有小图能在 1 天内跑完。
- App-BM+：从较大的 \(\mu\) 开始试，不满足停止条件就减小 \(\mu\)。基数保证略松，速度实用。定理 14 的因子满足 \(\lvert B\rvert\le(4\ln n+4)\lvert B_2^\circ\rvert=O(\ln n)\lvert OPT\rvert\)，和次模覆盖的最优阶相同。允许阈值有小幅违反（bicriteria）。作者认为这是影响力分析里第一个对「非次模覆盖」给出实用近似的算法。

经验取舍（论文 Remark）：中小图上 App-BM+ 又少又快，优先用它。部分大图上它仍然更少，但 Heur-BM 更快；要小集合用 App-BM+，要时间用 Heur-BM。

## 6. 实验（第 5 节）

实现 C++，Xeon 2.10 GHz，512 GB。9 个 SNAP 图：EmailCore、Facebook、Wiki-Vote、EmailAll、DBLP、Twitter、Stanford、Youtube、Pokec。最大 Pokec 约 163 万节点、3062 万边。

默认：

- IMIN：\(\epsilon=0.2\)，\(\beta=\gamma=0.1\)，\(\delta=1/n\)，\(\lvert S\rvert=10\)，\(k=100\)。种子从影响力 top-200 里随机抽。AG/GR 每轮 10000 个实现。
- BM：\(\lvert S\rvert=10\)，\(\eta=0.2\,\hat I_G(S)\)。Base 固定 1000 棵树。Heur-BM 的 \(\theta_h=10000\)，\(\gamma=0.1\)。App-BM 的 \(\mu=10^{-2}\)。
- 影响力用 \(10^5\) 次蒙特卡洛评估。每个设置重复 10 次。超过 24 小时记 INF。

IMIN 结果：

- SandIMIN / SandIMIN- 在所有图上快于 AG/GR，最多约两个数量级。大图上差距缩小，因为它们的样本数随图和理论保证增长，AG/GR 每轮样本数固定且无保证。
- 削减人数和 AG/GR 接近。AG/GR 在 Youtube、Pokec 上没有结果。
- 可计算的近似比下界：\(\epsilon=0.2,\gamma=0.1\) 时全部超过 20%；\(\epsilon=0.1,\gamma=0.05\) 时超过 30%，接近 \(1-1/e-\epsilon\)。

BM 结果：

- 速度：Base 最快；App-BM 最慢且常常一天跑不完；App-BM+ 对 \(\lvert S\rvert\) 和 \(\eta\) 相对稳。
- 人数：App-BM+ 稳定少于 Heur-BM。
- 阈值满足比：Base 多数低于 1；Heur-BM 经常明显高于 1；App-BM+ 在少数图上低于 1，与 bicriteria 的设定一致。

## 7. 和本仓库的关系（读完后应记住的判断）

本仓库的阻断协议见 `docs/tgn_blocking_report.md`：weibo22 是转发**树**，观察窗口 \(t_0\) 之后做子树切除，指标 `block_rate`。这和论文有三处不能直接等同。

1. 图上 \(D(B)\) 是 IC 随机传播的期望削减。weibo22 上未来转发已经发生，评价是实现树上的反事实切除，没有边概率、没有多父节点。树上从根到任意节点只有一条路，组合效应消失，\(D_L=D=D_U\)，夹心界不再提供额外信息。支配树退化成这棵转发树本身。
2. 论文假设全图和 \(p(u,v)\) 已知，阻断时能看到整个静态图。本仓库在 \(t_0\) 只能看到已经出现的节点，未来子树是要预测的量。MiniTGN 没超过观察出度，缺的是这个预测，不是缺一个静态图上的近似算法。
3. 论文没有强化学习。它给出的是预算型 IMIN 和阈值型 BM 的问题定义、边际增益贪心，以及「非次模时贪心没有基数保证、但实验里往往还行」。

可以直接搬进本仓库的部分：

- 把现有 `block_rate` 明确写成 IMIN；另报 BM：给定削减比例 \(\eta\)，最少要阻断几个观察节点。引理 2 说明只报 \(k=5\) 的 block_rate 覆盖不了「更少阻断达到同样效果」。
- Heur-IMIN 的打分 \( \Pr[\text{激活}]\times\text{出度} \) 与现有 `outdegree_obs`、`followers` 是同一族结构基线。观察出度已经是强基线，新模型必须超过它。
- 选中一个节点后，其下游候选的边际增益应降为 0。现在的 top-\(k\) 预测会把同一条链上的祖先和后代一起选上，这正是论文强调的边际重算。树上不必上 CP 采样，选完之后把观察子树里的候选标成冗余即可。
- 若以后把约 286 万用户收成跨级联有向图，再考虑调用作者仓库里的 SandIMIN / App-BM+，边概率仍用 \(1/\mathrm{in\text{-}degree}\)。那是另一套指标，不能和树协议的 `block_rate` 混在一张表里。
- 作者点名的下一步是 time-aware 扩散。这和本仓库的 1h/6h/24h 观察窗口是同一方向，但是他们的未来工作，不是本文已完成的算法。
