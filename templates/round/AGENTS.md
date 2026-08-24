# Round 工作目录

你在一个 TileOPs round 中工作。当前目录是 `/workspace/round`，TileOPs worktree
是 `/workspace/tileops`。所有 HIR、实验脚本、日志、profile 和报告都留在当前
目录；只把最终 kernel/dispatch 修改写入 TileOPs worktree。

## 约定

1. 先读 `brief.md` 和 `knowledge/tilelang.md`。
2. TileFoundry 的能力直接询问已安装的 `tilefoundry` 命令，不读 TileFoundry 源码。
3. 需要 GPU、TileLang 或 TileFoundry runtime 的命令直接运行；当前 shell 已经在
   TileOPs Docker 中，不需要 `tileops-run` 或宿主路径包装。
4. 不在容器里 commit、push 或创建 PR。完成后把宿主 worktree 的 diff 交给人处理。
5. 每个重要结论都留在 `evidence/` 或 `report.md`，不要只写在对话里。

TileOPs 项目的 `CLAUDE.md`、manifest、workload、reference 和测试是生产契约，
从 `/workspace/tileops` 读取。round 目录是实验工作区，不是要提交到 TileOPs 的
代码树。
