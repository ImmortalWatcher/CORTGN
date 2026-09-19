# weibo22

微博谣言级联树（Ma 等人常用设定）。本目录随 CORTGN 仓库提供，供阻断与反事实实验使用。

```text
events.txt                 eid + label（0 非谣言 / 1 谣言）
posts/Weibo/<eid>.json     一棵转发树，节点含 parent、t、uid、followers_count 等
```

4664 棵树，约 381 万节点。盘点见 `scripts/weibo22_inventory.json`，实验说明见 `docs/tgn_blocking_report.md`。
