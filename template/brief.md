# {{TASK}}

本轮是给开源算子库 [tile-ai/TileOPs](https://github.com/tile-ai/TileOPs) 贡献一个 GPU
kernel：按公开 issue 描述实现或优化一个算子，用仓库自带的 reference 验证数值，再和公开
的对照实现比性能，最后按上游规范提 PR。代码、证据和 PR 全部进公开仓库。工具链是
[tile-ai/TileFoundry](https://github.com/tile-ai/TileFoundry)——一个分析张量程序开销、
选择硬件指令并下降成 TIR 的编译器前端，同样开源。

**全程必须使用 TileFoundry：实际编写、分析并测量多个不同的 Sharded/Placed HIR，
由最终保留的 HIR 导出 production kernel 的结构改动，并把该 HIR 原样放进 PR；只调
config、launch、tile size 或 stage 不得开 PR。TileFoundry 能力阻塞时记录 finding 并
继续仍可执行的 workflow，不得绕过 TileFoundry 另写实现，也不得因此停手。阻塞的例子
按原样保留成最小复现，不要改小到能过为止。**

> {{PROMPT}}

| 环境 | 值 |
| --- | --- |
| branch | `{{BRANCH}}` |
| TileOPs base | `{{TILEOPS_BASE}}` |
| TileFoundry base | `{{TILEFOUNDRY_BASE}}` |

## 开发流程

1. 通读 `tilefoundry tutorial` 列出的所有页，从 manifest、Op、workload、reference、测试和
   benchmark 确认真实 contract；第一个 production runtime twin 正确前不要看 incumbent
   kernel body。
2. runtime twin 必须实际调用拟提交的 production TileLang kernel，并通过
   `tilefoundry check`；不得用 Torch、evaluator 或 detached implementation 代替。
3. 至少实测10个不同的 Sharded/Placed HIR。每个都保存 `tilefoundry analyze` 原始 JSON、
   placement、hypothesis、latency 和 kept/rejected 结论，再据此选择 final HIR。
4. final HIR 定下来后用 schedule 把指令选择定死：`tilefoundry schedule candidates` 列出
   每个未调度点位可用的指令，`tilefoundry schedule facts --target` 核对该 target 对选中
   指令的约束，在 HIR 里用 `tf.schedule` 写下选择，`tilefoundry schedule finalize` 落成
   TIR。authored form 以 `tilefoundry spec schedule` 为准。
5. production kernel 按这份 TIR 写，指令、buffer 深度和 operand 布局不得凭空发明；重新
   做 correctness、全部 primary workload benchmark、最强可运行 external baseline 和 profile。

当前目录是 `/workspace/round`，生产代码只写 `/workspace/tileops`。命令直接运行；
TileOPs 已 editable install，不拼 `PYTHONPATH`，不改变 public Op、manifest、workload、
reference、benchmark 或评估路径。本轮若是新增算子，可以在 `src/tileops/manifest/spec/`
下加一条自己的 entry，但不得改动已有 entry。

## TileFoundry 源码

`/workspace/tilefoundry` 是 TileFoundry 在 base `{{TILEFOUNDRY_BASE}}` 上的一份可写
checkout，容器里的 `tilefoundry` 就是它的 editable 安装——改完立刻生效，不用重装。
`tilefoundry tutorial`、`spec` 和 `tests/fixtures/schedule/` 都直接读这份源码。

挡路的 TileFoundry bug 自己修：先把最小复现留在 `work/blocked/<finding-id>/`，再改
`/workspace/tilefoundry` 并跑该模块自己的 pytest，两者都写进 `findings.json`。修完自己
收尾——在 `/workspace/tilefoundry`（已在分支 `foundry/{{TASK}}` 上）提交，push 到
`origin`，用挂载好的 `gh` 先开 issue，再对 `tile-ai/TileFoundry` 开 PR 并引用该 issue。

**动那个仓库之前先读 `/workspace/tilefoundry/CONTRIBUTING.md`**，commit、issue 和 PR 的
标题格式、body 分节、分支命名一律以它为准——它不在你的工作目录里，不会自动进上下文。
轮次结束还会把整份改动导出成 `foundry.patch` 留档。交付的仍然是 TileOPs 的算子，
TileFoundry 只是写算子的工具。

## Round 交付物

- `work/final_hir.py`（必须是带 `tf.schedule` 的那份）、`work/runtime_twin.py` 和至少
  一个不同 placement 的候选 HIR；
- `schedule finalize` 产出的 TIR，以及 candidates 和 facts 报告原文；
- `provenance.json`：记录 base、classification、final/runtime/kernel 路径、check、
  iterations、decisions、correctness、benchmark 和 profile；
- 每个 iteration 的 analyze JSON、正数 `measured_ms`、placement、hypothesis 和 verdict；
- `findings.json`，无 finding 时写 `{"findings": []}`；每条 finding 的 `reproducer` 指向
  `work/blocked/<finding-id>/` 里那个能单独运行的最小 HIR 或 TIR 文件，同目录放原样的
  命令和它的原始输出；
- correctness、benchmark、profile、baseline 原始证据，以及 `report.md`、`pr-title.txt`、
  `pr-body.md`。

实验脚本和证据留在 round，production diff 只留在 TileOPs worktree。

## PR

只有 candidate correctness 通过、全部 primary workload 完成同 contract 对比、相对
incumbent 有改进且 production kernel 有结构改动时才开 performance PR；否则在
`report.md` 记为 `no improvement`，不创建或保留 PR。

新增算子这一轮没有 incumbent：对照物换成 reference（正确性）和最强可运行的 external
baseline（性能），Performance 表去掉 incumbent 那一列，开 PR 的条件是 correctness 通过
且不慢于 baseline；慢于 baseline 就记 `no improvement`，不开 PR。

commit 和 PR title 使用：

```text
[Perf][foundry][<Scope>] <imperative description of the kernel change>
```

完成 commit 后运行 `open_pr.sh`。第一次调用会检查 artifacts 并停下来要求语义自检；
重新阅读实际 final HIR、analyze/measure 证据和 production diff 后，才运行第二条命令。
不得直接调用 `gh pr create`/`gh pr edit` 绕过它。使用全局 Git 身份，不加额外署名；
push 到 origin，不 merge，持续处理 CI/review。

```bash
./open_pr.sh
./open_pr.sh --reviewed
```

base 更新后 rebase、重新验证并 `--force-with-lease`。本轮若改过 TileFoundry，在
`report.md` 里写上那条 issue 和 PR 的编号，并说明 TileOPs 这条 PR 是否依赖它先合。

两个 PR 都开完后停下来报告链接，不要自行结束轮次：容器、worktree 和这个会话都留着，
CI 和 review 在同一个会话里继续跟。

### 完整 PR 模板

公开 PR body 严格只含下面四节：

````markdown
## Summary

- <production kernel 的关键改动及审查边界>
- <保持不变的 public Op、fallback、输出语义或兼容性>
- <candidate、incumbent、external baseline 和 primary workload 覆盖>

## TileFoundry Description

```python
@module(
    entry="<entry>",
    target=<target>,
    <final_hir.py 中的其他 module 参数>,
)
class <ModuleName>:
    @func
    def <entry>(<完整参数和类型>) -> <完整返回类型>:
        <原样填写 final_hir.py 中实际 analyze、实测并 finalize 通过的完整 HIR body，
         含全部 tf.schedule 指令选择>
```

## Performance

Operator: `<准确的 public TileOPs Op 名称>`

| Environment | Value |
| --- | --- |
| image | `<runner image>` |
| digest | `sha256:<image digest>` |
| gpu | `<GPU 型号>` |
| driver | `<driver version>` |
| cuda | `<CUDA version>` |
| torch | `<PyTorch version>` |
| tilelang | `<TileLang version/commit>` |
| <external stack> | `<external baseline version/commit>` |
| timer | `<TileOPs benchmark 使用的 timer>` |

Method: <本轮使用的 TileOPs benchmark。>

Ratio in comparator columns: implementation / candidate. &#x1F7E2; > 1 means the candidate is faster;
&#x1F534; <= 1 means it is not.

| Workload | <dim 1> | <dim 2> | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms)<br>/ candidate（新增算子无此列） | <External baseline> (ms)<br>/ candidate |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| <primary workload 1> | <value> | <value> | <dtype> | <candidate median> | **<incumbent median><br><marker>&nbsp;<incumbent/candidate ratio>x** | <external median><br><marker>&nbsp;<external/candidate ratio>x |
| <每个 primary workload 各一行> | ... | ... | ... | ... | ... | ... |
| geometric mean |  |  |  | <candidate geomean> | **<incumbent geomean><br><marker>&nbsp;<incumbent/candidate ratio>x** | <external geomean><br><marker>&nbsp;<external/candidate ratio>x |

## Result And Limitations

**<measured SOTA | improvement without SOTA>**

- <用逐行结果和 geometric mean 解释 classification>
- <列出仍慢于 incumbent 或 external baseline 的 workload>
- <specialization/fallback 边界和剩余风险>
- <本轮确认的 TileFoundry、lowering、runtime 或性能限制>
````

公开 title/body 不放 correctness/reproducer 命令、内部文件名或日志、本地绝对路径，
也不声明未经测量支持的 SOTA、通用性或 provenance；这些证据留在 round。
