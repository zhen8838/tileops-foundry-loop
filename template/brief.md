# {{TASK}}

**全程必须使用 TileFoundry：实际编写、分析并测量多个不同的 Sharded/Placed HIR，
由最终保留的 HIR 导出 production kernel 的结构改动，并把该 HIR 原样放进 PR；只调
config、launch、tile size 或 stage 不得开 PR。TileFoundry 能力阻塞时记录 finding 并
继续仍可执行的 workflow，不得绕过 TileFoundry 另写实现，也不得因此停手。**

> {{PROMPT}}

| 环境 | 值 |
| --- | --- |
| branch | `{{BRANCH}}` |
| TileOPs base | `{{TILEOPS_BASE}}` |
| TileFoundry wheel | `{{TILEFOUNDRY_COMMIT}}` |

## 开发流程

1. 运行 `tilefoundry tutorial optimize`，从 manifest、Op、workload、reference、测试和
   benchmark 确认真实 contract；第一个 production runtime twin 正确前不要看 incumbent
   kernel body。
2. runtime twin 必须实际调用拟提交的 production TileLang kernel，并通过
   `tilefoundry check`；不得用 Torch、evaluator 或 detached implementation 代替。
3. 至少实测两个不同的 Sharded/Placed HIR。每个都保存 `tilefoundry analyze` 原始 JSON、
   placement、hypothesis、latency 和 kept/rejected 结论，再据此选择 final HIR。
4. `tilefoundry schedule` 可用时运行并保存结果；若它阻塞，写入 `findings.json` 和最小
   reproducer，然后继续 analyze、实测和 kernel 优化。schedule 失败本身不是停手理由。
5. 根据 final HIR 和分析证据改写 production kernel，重新做 correctness、全部 primary
   workload benchmark、最强可运行 external baseline 和 profile。

当前目录是 `/workspace/round`，生产代码只写 `/workspace/tileops`。命令直接运行；
TileOPs 已 editable install，不拼 `PYTHONPATH`，不改变 public Op、manifest、workload、
reference、benchmark 或评估路径。TileFoundry 能力以已安装的 `tilefoundry` 命令为准。

## Round 交付物

- `work/final_hir.py`、`work/runtime_twin.py` 和至少一个不同 placement 的候选 HIR；
- `provenance.json`：记录 base、classification、final/runtime/kernel 路径、check、
  iterations、decisions、correctness、benchmark 和 profile；
- 每个 iteration 的 analyze JSON、正数 `measured_ms`、placement、hypothesis 和 verdict；
- `findings.json`，无 finding 时写 `{"findings": []}`；
- correctness、benchmark、profile、baseline 原始证据，以及 `report.md`、`pr-title.txt`、
  `pr-body.md`。

实验脚本和证据留在 round，production diff 只留在 TileOPs worktree。

## PR

只有 candidate correctness 通过、全部 primary workload 完成同 contract 对比、相对
incumbent 有改进且 production kernel 有结构改动时才开 performance PR；否则在
`report.md` 记为 `no improvement`，不创建或保留 PR。

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

base 更新后 rebase、重新验证并 `--force-with-lease`。

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
        <原样填写 final_hir.py 中实际 analyze 并实测保留的完整 HIR body>
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

| Workload | <dim 1> | <dim 2> | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms)<br>/ candidate | <External baseline> (ms)<br>/ candidate |
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
