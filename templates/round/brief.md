# {{OPERATOR}}

| 字段 | 值 |
| --- | --- |
| Scope | {{SCOPE}} |
| Same-contract baseline | {{BASELINE}} |
| TileOPs base | `{{TILEOPS_BASE}}` |
| TileFoundry requested commit | `{{TILEFOUNDRY_COMMIT}}` |
| Round slug | `{{SLUG}}` |

## 目标

在不改变 public Op、manifest、workload、reference、benchmark 或评估路径的前提
下，为这个 operator 做一个可验证的 TileLang kernel，并给出与 incumbent 和
最强同契约 baseline 的实测比较。

## 起点

先发现 `/workspace/tileops` 中真实的 manifest、Op、workload、reference、测试和
benchmark。不要先看 incumbent kernel body；先记录一个正确的 production runtime
twin，再研究现有实现。

使用已安装命令了解 TileFoundry：

```bash
tilefoundry tutorial
tilefoundry spec
tilefoundry models
tilefoundry analyze ...
tilefoundry schedule ...
tilefoundry check ...
```

所有命令都在 Docker 中运行。`knowledge/tilelang.md` 是历史测量笔记，必须结合
当前 wheel 和当前 CUDA/Torch/TileLang 版本复核；不能把它当作 TileFoundry API
文档。

## 交付

工作完成时，round 目录至少包含：

- authored HIR、production runtime twin 和对应的 `tilefoundry check` 结果；
- analyze/schedule 结果、从结果到 kernel 选择的决定记录；
- correctness、benchmark、profile 和 baseline 原始证据；
- `report.md`，说明结果、限制和未解决的 TileFoundry 行为。

不要在这里 commit、push 或开 PR。宿主会审核 `/workspace/tileops` 的 diff 并完成
Git/PR 流程。
