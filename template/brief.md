# {{TASK}}

使用已安装的 TileFoundry 完成以下任务：

> {{PROMPT}}

| 环境 | 值 |
| --- | --- |
| branch | `{{BRANCH}}` |
| TileOPs base | `{{TILEOPS_BASE}}` |
| TileFoundry wheel | `{{TILEFOUNDRY_COMMIT}}` |

## 工作约束

- 当前目录是 `/workspace/round`，生产代码只写入 `/workspace/tileops`。
- 先从 manifest、Op、workload、reference、测试和 benchmark 确认真实 contract；
  第一个 production runtime twin 正确前不要看 incumbent kernel body。
- TileFoundry 的能力直接询问已安装的 `tilefoundry` 命令，不读宿主 TileFoundry
  checkout，也不维护它的静态命令清单。
- CUDA、TileLang、TileFoundry 和 GPU 命令直接运行。TileOPs 已 editable install，
  不使用宿主包装器，也不额外拼接 `PYTHONPATH`。
- 修改文件使用 Pi 自带的 `edit`/`write` 工具。
- 不改变 public Op、manifest、workload、reference、benchmark 或评估路径。
- 验证完成后按 TileOPs 项目规则 commit、push、创建 PR 并跟进 CI；不要 merge。

`knowledge/tilelang.md` 只记录明确版本上复现过的事实，所有结论都要用当前环境
复核。

## 交付物

round 目录最终至少包含：

- authored HIR、production runtime twin，以及对应的 `tilefoundry check` 原始结果；
- analyze/schedule 结果和从这些结果到 kernel 决策的记录；
- correctness、benchmark、profile、最强同 contract baseline 的原始证据；
- `report.md`，说明结果、限制和发现的 TileFoundry 问题。

生产 diff 只留在 `/workspace/tileops`，实验脚本和证据只留在当前 round。

## TileOPs PR 规范

只有 correctness 通过且相对 incumbent 有可审查的性能改进时才创建 performance PR。
commit 和 PR title 使用：

```text
[Perf][foundry][<Scope>] <imperative description>
```

public PR body 严格只含以下四节：

1. `Summary`
2. `TileFoundry Description`：一个 Python code block，内含最终的完整 `@module`
   class；不放 import、文件名、路径或单独打印的 entrypoint
3. `Performance`：环境、测量方法和全部 primary workloads；candidate 列只放 latency，
   每个 comparator cell 第一行放 latency、第二行放 `implementation / candidate`；
   incumbent cell 加粗，并明确比值大于 1 才表示 candidate 更快
4. `Result And Limitations`：结论必须是 `measured SOTA` 或
   `improvement without SOTA`，并如实列出仍落后的 workload 和适用边界

correctness/reproducer 命令、内部 artifact、宿主路径和 round 路径只留在
`report.md`，不得进入 public PR body。push 到 fork 后，用 `gh pr create` 向
`tile-ai/TileOPs:main` 创建 PR，并持续处理 CI/review；不要 merge。
