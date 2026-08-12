# Round 2 blind-phase read audit

Blind phase started no later than `2026-08-12T01:49:34+08:00`.

The incumbent implementation `src/tileops/kernels/gemm_w4a16.py`, generated copies,
compiler caches, and import-source/disassembly views of that kernel were not read.

## Source and contract read commands

```bash
sed -n '1,260p' $TRIAL_SOURCE/round-2-gemm-w4a16/brief.md
sed -n '1,260p' CLAUDE.md
git status --short --branch
rg --files -g 'CLAUDE.md' -g 'AGENTS.md' -g '*RULES*' -g '*rules*'
sed -n '261,620p' $TRIAL_SOURCE/round-2-gemm-w4a16/brief.md
sed -n '1,260p' .claude/domain-rules/ops-design.md
sed -n '1,260p' .claude/domain-rules/benchmark.md
sed -n '1,260p' .claude/domain-rules/testing-budget.md
sed -n '1,300p' docs/design/trust-model.md
git rev-parse HEAD && git merge-base HEAD main && git -C $HOST_HOME/TileFoundry rev-parse HEAD
date --iso-8601=seconds
$HOST_HOME/foreman/local/tileops-container.sh python -c "from tileops.manifest import load_manifest, load_workloads; import pprint; m=load_manifest(); pprint.pp(m['ops']['GemmW4A16Op']); print('WORKLOADS'); pprint.pp(load_workloads('GemmW4A16Op'))"
rg -n "GemmW4A16|quantize_weight_int4|w4a16|Marlin|marlin" workloads/gemm.py src/tileops/ops/gemm.py tests/ops/test_gemm.py benchmarks/ops/bench_gemm.py benchmarks/benchmark_base.py src/tileops/manifest/gemm.yaml
rg --files $HOST_HOME/TileFoundry | rg '(^|/)(README|.*tutorial|.*spec|examples|docs|pyproject|.*\\.md$)' | head -200
$HOST_HOME/TileFoundry/.venv/bin/tilefoundry --help
sed -n '1,260p' docs/design/testing.md
sed -n '90,180p' src/tileops/manifest/gemm.yaml && sed -n '90,190p' workloads/gemm.py && sed -n '320,470p' src/tileops/ops/gemm.py
sed -n '1,45p' tests/ops/test_gemm.py && sed -n '145,240p' tests/ops/test_gemm.py && sed -n '285,330p' tests/ops/test_gemm.py
sed -n '1,215p' benchmarks/ops/bench_gemm.py && sed -n '220,380p' benchmarks/ops/bench_gemm.py
sed -n '1,280p' $HOST_HOME/TileFoundry/README.md && sed -n '1,320p' $HOST_HOME/TileFoundry/docs/spec/hir.md
sed -n '1,300p' $HOST_HOME/TileFoundry/docs/spec/cli.md && sed -n '1,260p' $HOST_HOME/TileFoundry/docs/spec/evaluator.md
sed -n '1,300p' $HOST_HOME/TileFoundry/docs/tutorial/index.md && sed -n '1,320p' $HOST_HOME/TileFoundry/docs/tutorial/optimize.md && sed -n '1,320p' $HOST_HOME/TileFoundry/docs/tutorial/migrate.md
$HOST_HOME/foreman/local/tileops-container.sh python -c "from tileops.manifest import load_manifest, load_workloads; import pprint; m=load_manifest(); print(type(m), list(m)[:5]); pprint.pp(m['GemmW4A16Op']); print('WORKLOADS'); pprint.pp(load_workloads('GemmW4A16Op'))"
```

The first manifest command failed with `KeyError: 'ops'`; the second established that
`load_manifest()` directly returns the op-name mapping.

```bash
$HOST_HOME/TileFoundry/.venv/bin/tilefoundry models && $HOST_HOME/TileFoundry/.venv/bin/tilefoundry models qwen3_5_35b_a3b --source
$HOST_HOME/TileFoundry/.venv/bin/tilefoundry check --help && $HOST_HOME/TileFoundry/.venv/bin/tilefoundry analyze --help && $HOST_HOME/TileFoundry/.venv/bin/tilefoundry schedule --help
rg -n "class (Cast|Reshape|BroadcastTo|Transpose|Matmul|MatMul|Stack|FloorDiv|Mod)|def (cast|reshape|broadcast_to|transpose|matmul|stack|floor_div|mod)" $HOST_HOME/TileFoundry/src/tilefoundry/ir $HOST_HOME/TileFoundry/src/tilefoundry/frontend $HOST_HOME/TileFoundry/src/tilefoundry/evaluator 2>/dev/null
rg -n "@func|Tensor\\[|tf\\.(cast|reshape|broadcast_to|transpose|matmul|stack|floor_div|mod)" $HOST_HOME/TileFoundry/src/tilefoundry/models $HOST_HOME/TileFoundry/tests -g '*.py' | head -240
if [ -f src/tileops/kernels/rms_norm.py ]; then sed -n '1,300p' src/tileops/kernels/rms_norm.py; fi
if [ -f src/tileops/kernels/activation.py ]; then sed -n '1,260p' src/tileops/kernels/activation.py; fi
sed -n '1,280p' src/tileops/kernel.py
sed -n '440,490p' src/tileops/ops/gemm.py && sed -n '45,205p' benchmarks/ops/bench_gemm.py && sed -n '320,375p' benchmarks/ops/bench_gemm.py
```

The attempted `src/tileops/kernel.py` read failed because that path does not exist.

```bash
sed -n '1,35p' src/tileops/ops/gemm.py
$HOST_HOME/foreman/local/tileops-container.sh python -c "from tileops.manifest import load_manifest; m=load_manifest(); print('\\n'.join(sorted({v.get('source',{}).get('kernel','') for k,v in m.items() if v.get('family') != 'gemm' and v.get('source',{}).get('kernel')})))"
sed -n '1,240p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor/stack.py && sed -n '1,220p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor/cast.py && sed -n '1,240p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor/reshape.py && sed -n '1,220p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/nn/matmul.py
rg -n "^(def|class) |^[a-zA-Z_][a-zA-Z0-9_]* =|__all__" $HOST_HOME/TileFoundry/src/tilefoundry/dsl/tf.py $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor -g '*.py'
sed -n '1,240p' $HOST_HOME/TileFoundry/tests/evaluator/test_eval_core.py && sed -n '1,180p' $HOST_HOME/TileFoundry/tests/models/qwen3_5_35b_a3b/model.py
sed -n '1,320p' src/tileops/kernels/kernel_base.py && sed -n '1,340p' src/tileops/kernels/norm/rms_norm.py && sed -n '1,320p' src/tileops/kernels/reduction/reduce.py
sed -n '1,280p' $HOST_HOME/TileFoundry/src/tilefoundry/dsl/tf.py && sed -n '1,240p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor/__init__.py && rg -n "Floor|Mod|Div|Broadcast" $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir -g '*.py'
sed -n '1,340p' $HOST_HOME/TileFoundry/docs/spec/runtime.md && rg -n "runtime_module|@.*runtime|tilefoundry check" $HOST_HOME/TileFoundry/tests $HOST_HOME/TileFoundry/docs -g '*.py' -g '*.md' | head -220
sed -n '1,260p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/nn/matmul.py && sed -n '1,220p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor/transpose.py && rg -n "tf\\.matmul|matmul\\(" $HOST_HOME/TileFoundry/tests/models/qwen3_5_35b_a3b/model.py | head -40
$HOST_HOME/foreman/local/tileops-container.sh python -c "import torch, tilelang; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability()); print('torch', torch.__version__); print('tilelang', tilelang.__version__)" && $HOST_HOME/foreman/local/tileops-container.sh sh -lc 'nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader'
sed -n '1,240p' $HOST_HOME/TileFoundry/src/tilefoundry/dsl/tf.py
rg -n "class (Floor|Mod|Div|Binary|Broadcast)|register_op|__all__" $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/elementwise $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/tensor -g '*.py' 2>/dev/null
sed -n '1,120p' $HOST_HOME/TileFoundry/tests/fixtures/placed/square_twin.py && sed -n '1,100p' $HOST_HOME/TileFoundry/tests/fixtures/placed/fused_twin.py && sed -n '1,120p' $HOST_HOME/TileFoundry/tests/fixtures/placed/square_cpu_runtime.py
rg -n "@tilelang\\.jit|with T\\.Kernel|T\\.alloc_fragment|T\\.reduce_sum|@torch\\.library\\.custom_op" src/tileops/kernels/norm/rms_norm.py src/tileops/kernels/reduction/reduce.py | head -160
rg --files $HOST_HOME/TileFoundry/src/tilefoundry/dsl/tf $HOST_HOME/TileFoundry/src/tilefoundry/dsl | head -160
sed -n '1,260p' $HOST_HOME/TileFoundry/src/tilefoundry/dsl/tf/__init__.py 2>/dev/null; rg -n "floor|mod|broadcast|repeat|stack|def __" $HOST_HOME/TileFoundry/src/tilefoundry/dsl $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir -g '*.py' | head -240
sed -n '70,145p' src/tileops/kernels/reduction/reduce.py && sed -n '155,235p' src/tileops/kernels/reduction/reduce.py && sed -n '800,940p' src/tileops/kernels/reduction/reduce.py
sed -n '1,240p' $HOST_HOME/TileFoundry/src/tilefoundry/ir/hir/elementwise.py 2>/dev/null; rg -n "name=\\\"(mod|floor_div|floordiv|remainder|bitwise)\\\"|class .*Mod|torch\\.(remainder|floor_divide)" $HOST_HOME/TileFoundry/src/tilefoundry -g '*.py'
rg -n "class BenchmarkBase|def profile|CUPTI|cupti|do_bench|warmup|repeat|rep" benchmarks/benchmark_base.py benchmarks/conftest.py
sed -n '1,240p' benchmarks/benchmark_base.py && sed -n '240,520p' benchmarks/benchmark_base.py
sed -n '520,670p' benchmarks/benchmark_base.py && sed -n '700,810p' benchmarks/benchmark_base.py
sed -n '1,120p' benchmarks/ops/bench_gemm.py && sed -n '205,255p' benchmarks/ops/bench_gemm.py
```

Blind phase ended at exactly `2026-08-12T02:04:48+08:00`, after
`first-candidate.md` and all evidence it cites had been saved. No incumbent source,
generated copy, compiler cache, import-source view, or disassembly was read before
this timestamp.
