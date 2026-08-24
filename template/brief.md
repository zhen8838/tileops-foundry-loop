# {{TASK}}

**全程必须按照 TileFoundry workflow 开发；失败只能记录为 TileFoundry finding，
不得绕过 TileFoundry 另行实现。**

**config-only、launch-config-only、tile-size-only 或 tune-only 改动绝不构成
`[Perf][foundry]` PR。最终 TileFoundry Description 必须是经过分析和多轮比较后选定的
sharded、placed HIR；production diff 必须包含由该决策导出的结构性 kernel 编写或
schedule 实现改动。否则本轮只能记为 `no improvement`，不得创建或保留 performance PR。**

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
- 不改变 public Op、manifest、workload、reference、benchmark 或评估路径。
- 验证完成后按 TileOPs 项目规则 commit、push、创建 PR 并跟进 CI；不要 merge。
- 开发早期反复运行 `python check_round.py --tileops-repo /workspace/tileops`；它是
  `[Perf][foundry]` provenance 和 production diff 的机器门禁，不得绕过或修改。

`knowledge/tilelang.md` 只记录明确版本上复现过的事实，所有结论都要用当前环境
复核。

## TileFoundry Optimize 硬流程

1. 运行 `tilefoundry tutorial optimize`，从 manifest、Op、workload、reference、测试和
   benchmark 写出 authored HIR；第一个正确的 production runtime twin 之前不看
   incumbent kernel body。
2. runtime twin 必须通过 `tilefoundry check`，且实际调用本轮测量和拟提交的 production
   TileLang kernel，不得用 Torch/evaluator 或 detached implementation 代替。
3. 至少编写和实测两种**结构不同**的 sharded、placed HIR。每种 HIR body 都必须显式
   声明 target、`Mesh`、mesh-bound layout、`reshard`、`gmem` 和至少一种
   `smem`/`rmem`/`tmem`；只在 `@module(topologies=...)` 填数字不算 placement。
4. 每个候选都必须运行当前安装版本 `tilefoundry analyze --help` 暴露的全部 core
   analysis，以及带明确 topology 的 `tilefoundry schedule`，保存 JSON 原始结果并实测。
   至少一个 placement 被测量后拒绝；最终 kept placement 的 schedule 必须完成搜索，
   不得使用 `--first-plan`。
5. decision trace 必须逐项连接 `analysis_fact -> schedule_fact -> hir_change ->
   kernel_change`。最后根据这些证据改写 production `@T.prim_func`/`@T.macro` 的实现或
   schedule 结构，并重新做 correctness、benchmark、profile。
6. 仅改变 threads、warps、heads-per-program、block/tile size、stage 数等常量或 config，
   即使性能提高，也只能作为 rejected experiment，不能作为最终 production diff。

## 交付物

round 目录最终至少包含：

- `work/final_hir.py`、`work/runtime_twin.py` 和至少一个不同 placement 的候选 HIR；
- `provenance.json`：记录 `tileops_base`、classification、final/runtime/kernel 路径、
  check、iterations、decisions、correctness、benchmark 和 profile；
- 每个 iteration 的 analyze/schedule JSON、正数 `measured_ms`、placement、hypothesis 和
  `kept`/`rejected` verdict；
- `findings.json`，即使没有 finding 也必须是 `{"findings": []}`；
- correctness、benchmark、profile、最强同 contract baseline 的原始证据；
- `report.md`、`pr-title.txt` 和 `pr-body.md`。

生产 diff 只留在 `/workspace/tileops`，实验脚本和证据只留在当前 round。

## TileOPs PR 规范

### 创建条件

只有同时满足以下条件才创建 performance PR：

- candidate 通过 TileOPs contract、reference 和全部要求的 correctness 测试；
- candidate 是通过本轮 TileFoundry workflow 生成的 kernel，不把 incumbent-derived
  实现包装成 TileFoundry 产物；
- final HIR 是 sharded、placed form，并有至少一个被实测拒绝的不同 placement；
- base-to-head 含结构性的 production `@T.prim_func`/`@T.macro` 实现或 schedule diff；
  归一化掉常量后结构不变的 config/tile tuning 不合格；
- 在相同 contract 下，相对 TileOPs incumbent 有可审查的性能改进；
- candidate、incumbent 和所有可运行的最强 external baseline 已在全部 primary
  workloads 上使用 TileOPs 提供的 benchmark 实测；
- production diff 只包含可审查的 kernel 改动，没有更改 public Op、manifest、
  workload、reference、benchmark 或评估路径。

`no improvement` 是 `report.md` 可使用的本轮结论，但不得据此创建 performance PR。

### Title

commit 和 PR title 严格使用：

```text
[Perf][foundry][<Scope>] <imperative performance description>
```

- `Perf` 是变更类型，对应 `perf` label。
- `foundry` 是固定的小写来源标记，必须紧跟在类型后，对应 `foundry` label。
- `<Scope>` 是算子家族，例如 `GEMM`、`MoE`、`Mamba` 或 `FFT`；不得再次使用
  `foundry`，只使用字母、数字、下划线或连字符。
- description 使用简短祈使句，描述实际性能改动。
- description 必须描述 kernel/schedule 的结构性改动，不得写成 tune config/tile size。
- 目标分支必须支持三段式 title；不要为了通过旧 validator 擅自删除 `foundry`。

### 提交、创建 PR 与跟进

满足创建条件时，工作不得停在 `report.md` 或本地 commit；必须自行通过
`open_pr.sh` push、创建或更新 PR，并持续处理 CI/review。`open_pr.sh` 会先执行
`check_round.py`；不得直接运行 `gh pr create`/`gh pr edit` 绕过它。使用全局 Git
身份，不加**额外署名**；push 到 origin，不 merge。

```bash
./open_pr.sh
```

CI 失败先读失败 step；base 更新后 rebase、重新验证并 `--force-with-lease`。

### 完整 PR 模板

公开 PR body 严格只含下面四节，按原顺序填写，不增加 Correctness、Reproduce 或
Artifacts 等公开章节：

````markdown
## Summary

- <说明 production kernel 的关键改动及其审查边界>
- <说明保持不变的 public Op、fallback、输出语义或兼容性>
- <说明覆盖了 candidate、incumbent、external baseline 和全部 primary workloads>

## TileFoundry Description

```python
@module(
    entry="<entry>",
    target=<target>,
    <本轮最终 HIR 的其他 module 参数>,
)
class <ModuleName>:
    @func
    def <entry>(<完整参数和类型>) -> <完整返回类型>:
        <final_hir.py 中经过比较选定的完整 sharded、placed HIR body；必须含 Mesh、
        mesh-bound layout、reshard 和显式 storage placement>
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

Method: <准确写明本轮使用的 TileOPs benchmark。>

Ratio in comparator columns: implementation / candidate. &#x1F7E2; > 1 means the candidate is faster;
&#x1F534; <= 1 means it is not.

| Workload | <dim 1> | <dim 2> | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms)<br>/ candidate | <External baseline> (ms)<br>/ candidate |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| <primary workload 1> | <value> | <value> | <dtype> | <candidate median> | **<incumbent median><br><marker>&nbsp;<incumbent/candidate ratio>x** | <external median><br><marker>&nbsp;<external/candidate ratio>x |
| <每个 primary workload 各一行，不省略退化或长尾行> | ... | ... | ... | ... | ... | ... |
| geometric mean |  |  |  | <candidate geomean> | **<incumbent geomean><br><marker>&nbsp;<incumbent/candidate ratio>x** | <external geomean><br><marker>&nbsp;<external/candidate ratio>x |

## Result And Limitations

**<measured SOTA | improvement without SOTA>**

- <用逐行结果和 geometric mean 准确解释 classification>
- <逐项列出仍慢于 incumbent 或 external baseline 的 workload>
- <说明 specialization/fallback 的适用边界和剩余风险>
- <说明本轮确认但未隐藏的 TileFoundry、lowering、runtime 或性能限制>
````

### 公开边界与提交前检查

correctness 和复现证据仍是开 PR 前的硬门禁，但只能留在 `report.md` 和 round
原始证据中。公开 title/body 不得包含：

- correctness/reproducer 命令、内部 artifact 或原始日志；
- `authored_hir.py` 等 Python 文件名，`Entrypoint:`、`Source:`；
- `/home/`、`/mnt/`、`/tmp/`、`/workspace/`、`/Users/`、`file://`、Windows
  盘符路径，或任何宿主/round 本地路径；
- 未经测量支持的 SOTA、通用性或 provenance 声明。

创建 PR 前逐项检查：

```text
correctness passed
  -> check_round.py PASS
  -> at least two materially different placed HIR iterations
  -> final production kernel has a structural body/schedule change
  -> all primary workloads present
  -> candidate + incumbent + strongest runnable external baseline complete
  -> all latencies positive and ratios/geomeans recomputed
  -> public body has exactly four ordered sections and no private evidence
  -> production diff and commit reviewed
  -> ./open_pr.sh
  -> follow CI/review; never merge
```
