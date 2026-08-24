# Round Workspace

这个目录由 `templates/round` 复制而来，是本轮 Agent 的完整工作区。

```text
/workspace/round/
  brief.md              本轮 operator、scope、baseline 和 commit
  AGENTS.md             工作约定
  knowledge/            可复用的 TileLang 测量笔记
  evidence/             原始日志、profile、reproducer 和环境记录
  work/                 HIR、runtime twin、实验脚本和临时输出
  report.md             最终内部报告
```

`/workspace/tileops` 是唯一的生产代码目标。容器提供 CUDA、TileLang、TileFoundry
wheel 和 TileOPs runtime；Git 和 PR 在宿主机完成。
