# TileLang: what previous work measured

Traps and conventions established by measurement or by reading generated CUDA.
Read before authoring a kernel. Entries tagged `[0.1.12]` were seen on
`tilelang 0.1.12 / torch 2.13.0+cu130`; this loop admits `0.1.11+cu132.gitafcebed1`,
so treat them as one-compile hypotheses and report the verdict when you check one.

## Hand-written warp specialization

| fact | what to do | build |
|---|---|---|
| Two sequential `T.ws` regions deadlock: `ThreadSync` emits a `__syncthreads()` the producer can never reach | nest both `T.ws` in the two arms of one `T.If`; the pass then summarises them as one statement. Written that way from the start on 0.1.11, one `__syncthreads()` (the barrier-init fence) is emitted and it works; the sequential form was not re-provoked there | [0.1.12] |
| `threads=384` hangs — two consumer warp groups make `T.gemm` need a block-wide sync inside a thread-dependent branch | one producer + one consumer, 256 threads; that shape confirmed working | [0.1.12], 256 confirmed [0.1.11] |
| **Register budget inside `T.ws` is honoured.** The generated CUDA carries `tl::warpgroup_reg_dealloc<24>()` and `tl::warpgroup_reg_alloc<240>()` | grep the wrappers, not the literal `setmaxnreg` -- a kernel whose budget *is* set contains no such string, which is probably what the earlier "silently dropped" reading was | corrected [0.1.11] |
| `T.Parallel` binds to the kernel's thread extent, not the enclosing guard: `T.If(tx >= 128)` masks half and leaves half uncomputed, symptom `'!!!!!!'` in the generated text, no error | scope with `T.ws(i)`, not with a thread guard | [0.1.12] |
| Allocations inside a guarded stage are invisible outside it | declare every ring and barrier set at kernel-body level, up front | [0.1.12] |
| Wrong barrier arrive counts hang the kernel and kill the CUDA context; the error surfaces at the *next* allocation | a hang whose traceback points at an unrelated `torch.empty` is this | [0.1.12] |

Barrier convention, read out of the automatic pass's own generated CUDA:

```text
full  barriers: init(1)    -- completed by the TMA transaction count
empty barriers: init(128)  -- all 128 consumer threads arrive
producer waits: empty[k % stages].wait(((k % (2*stages)) / stages) ^ 1)
consumer waits: full[k % stages].wait((k % (2*stages)) / stages)
```

Confirmed on 0.1.11, with one degree of freedom: `T.alloc_barrier([N] * stages)`
emits `bar[i].init(N)` verbatim and `N` must equal the number of threads that
call `T.mbarrier_arrive` on it -- `init(1)` when TileLang elects one thread for
the TMA (`tl_shuffle_elect<128>()` plus `expect_transaction(bytes)`), `init(128)`
when every thread in the guarded region arrives. Writing the phase as
`(t // stages) % 2` for the consumer and `^ 1` for the producer generates the
form above; equal expressions, checked as `bar[(t&1)].wait((t&3)>>1)`.

## Shared memory

| fact | what to do | build |
|---|---|---|
| One pool shared by all stages cost 194 KB; a staging pair per stage folded to 98 KB at 3 stages. With the automatic pass on, 2+ pipelined loops ask 288 KB against a 227 KB limit | declare per-stage pairs and let `MergeSharedMemoryAllocations` pack them | [0.1.12] |
| Non-monotonic in ring depth: `RQ=2` needed 328704 B, `RQ=8` needed 263168 B — depth sets the live range | do not shrink a ring to save memory; measure both ends | [0.1.12] |
| The allocator does not model branch exclusivity: `T.If` arms get separate addresses and stack (visible as `Ws4` / `Ws4_1`); `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` did not fix it | a two-armed kernel cannot afford the depths either arm affords alone | [0.1.12] |
| A swizzle-layout shared buffer cannot sit at an offset inside a bigger allocation. Five forms verified, all fail for TMA destinations: `T.Buffer(data=…, elem_offset=…)` (alias must cover the parent whole), `T.view` (no offset), `T.reshape(pool[a:b])` (no dtype), `T.match_buffer(pool[a:b])` (rank rule, then unbound data var), `match_buffer`+`view` (same) | give each ring its own buffer | [0.1.12] |
| A **leading stage axis on the allocation itself** is a different thing and works: `T.alloc_shared([stages, block_N, dim])` + `T.annotate_layout({Ks: make_swizzled_layout(Ks)})` + `T.tma_copy(..., Ks[st, :, :], barrier=...)` compiles and is correct | prefer this to carving offsets out of one pool | [0.1.11] |
| `MergeSharedMemoryAllocations` is optimal on a straight-line body: reproduced exactly by an independent packer (231424 B from 574208 B); hand placement measured 375296 B and worse | give it a straight-line body, do not place by hand | [0.1.12] |

## Layout, compile, diagnostics

| fact | what to do | build |
|---|---|---|
| Layout inference uses the total thread count, not the consumer subset: `threads=384` fails `warp_col_tiles must be divisible by 8, got 21`, `threads=512` finds no layout | one kernel cannot give stages different effective thread shapes; weigh that before fusing | [0.1.12] |
| `@tilelang.jit` breaks on a kernel with `T.If` or atomics: "`<last statement>` is not a callable object" | **does not reproduce on 0.1.11**: `@tilelang.jit` compiled a body with two `T.ws` in one `if`/`else`, and another with `T.atomic_add(return_prev=True)` plus `T.call_extern` `__threadfence`. `tilelang.compile(prim, pass_configs=...)` remains the escape hatch | [0.1.12], refuted [0.1.11] |
| The eager builder rewrites buffer assignment only inside the traced function: `reg[0] += T.shfl_xor(reg[0], off)` in a plain Python helper raises `TypeError: 'Buffer' object does not support item assignment` | write shuffle chains and any buffer mutation inline in the traced function or inside a `@T.macro` | [0.1.11] |
| `T.reduce_*` destinations must be read at their own extent: reading index 0 of a `[block_M]` fragment from a `T.Parallel` of another extent forces a replicated layout and collides -- `Layout may conflict with ReduceOp ... replicate_size=128` | run each epilogue loop over the fragment's own extent, guarding with `if i == 0` where only row 0 is wanted | [0.1.11] |
| `Immutable value 't6' is re-bound` = a stage body emitted twice | [0.1.12] |
| `asm` is a C++ keyword and silently breaks codegen as a buffer name | [0.1.12] |
| The compiled object reports nothing: `resource_usage` `{}`, `n_regs`/`n_spills` `None`, no `dynamic_smem_bytes` | get shared-memory size from a launch failure or by regexing generated CUDA for `buf_dyn_shmem + N` | [0.1.12] |

## Launch traps that were ours

| fact | fix | build |
|---|---|---|
| Graph capture failed ~1 run in 3, "Capture must end on the same stream it began on" — a tilelang kernel does not always launch on the stream it is called under | `torch.cuda.graph(g, stream=side)` | [0.1.12] |
| Any card but the default ran 26x slower with correct output — the launch stream comes from `torch.cuda.current_device()`, not from the tensors' device | pin with `CUDA_VISIBLE_DEVICES`; `tileops-run` already does, do not undo it | [0.1.12] |

## Confirmed expressible

`T.tma_copy` with mbarrier `full`/`empty` pairs (a hand-written producer/consumer
ring) · `T.ws(0)`/`T.ws(1)` scoping `T.Parallel` to a warp group ·
`T.sync_grid()` inside a CUDA graph · a consumer-only grid barrier from
`T.macro` + `T.atomic_add(return_prev=True)` + `T.atomic_load` + `T.While`
(validated 132/132 blocks) · per-pass control via `tilelang.compile(pass_configs=…)`.

## Growing this file

Establish something the next round would otherwise rediscover — a trap, a
convention read out of generated CUDA, a form that does not exist — and write it
under `TileLang notes` in the round report with what you did, what happened, and
the build. Promotion into this file is a human step. Keep entries one line:
claim, action, build.

First entries distilled from a prefill megakernel proof of concept whose
write-ups live outside this loop.
