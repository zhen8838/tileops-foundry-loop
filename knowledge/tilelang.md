# TileLang: what previous work measured

Traps and conventions established by measurement or by reading generated CUDA.
Read before authoring a kernel. Entries tagged `[0.1.12]` were seen on
`tilelang 0.1.12 / torch 2.13.0+cu130`; this loop admits `0.1.11+cu132.gitafcebed1`,
so treat them as one-compile hypotheses and report the verdict when you check one.

## Hand-written warp specialization

| fact | what to do | build |
|---|---|---|
| Two sequential `T.ws` regions deadlock: `ThreadSync` emits a `__syncthreads()` the producer can never reach | nest both `T.ws` in the two arms of one `T.If`; the pass then summarises them as one statement | [0.1.12] |
| `threads=384` hangs — two consumer warp groups make `T.gemm` need a block-wide sync inside a thread-dependent branch | one producer + one consumer, 256 threads | [0.1.12] |
| `T.dec_max_nreg` / `T.inc_max_nreg` inside `T.ws` are silently dropped (zero `setmaxnreg` emitted at 24/240 and 32/224) | do not budget registers on them; grep the generated source | [0.1.12] |
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

## Shared memory

| fact | what to do | build |
|---|---|---|
| One pool shared by all stages cost 194 KB; a staging pair per stage folded to 98 KB at 3 stages. With the automatic pass on, 2+ pipelined loops ask 288 KB against a 227 KB limit | declare per-stage pairs and let `MergeSharedMemoryAllocations` pack them | [0.1.12] |
| Non-monotonic in ring depth: `RQ=2` needed 328704 B, `RQ=8` needed 263168 B — depth sets the live range | do not shrink a ring to save memory; measure both ends | [0.1.12] |
| The allocator does not model branch exclusivity: `T.If` arms get separate addresses and stack (visible as `Ws4` / `Ws4_1`); `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` did not fix it | a two-armed kernel cannot afford the depths either arm affords alone | [0.1.12] |
| A swizzle-layout shared buffer cannot sit at an offset inside a bigger allocation. Five forms verified, all fail for TMA destinations: `T.Buffer(data=…, elem_offset=…)` (alias must cover the parent whole), `T.view` (no offset), `T.reshape(pool[a:b])` (no dtype), `T.match_buffer(pool[a:b])` (rank rule, then unbound data var), `match_buffer`+`view` (same) | give each ring its own buffer | [0.1.12] |
| `MergeSharedMemoryAllocations` is optimal on a straight-line body: reproduced exactly by an independent packer (231424 B from 574208 B); hand placement measured 375296 B and worse | give it a straight-line body, do not place by hand | [0.1.12] |

## Layout, compile, diagnostics

| fact | what to do | build |
|---|---|---|
| Layout inference uses the total thread count, not the consumer subset: `threads=384` fails `warp_col_tiles must be divisible by 8, got 21`, `threads=512` finds no layout | one kernel cannot give stages different effective thread shapes; weigh that before fusing | [0.1.12] |
| `@tilelang.jit` breaks on a kernel with `T.If` or atomics: "`<last statement>` is not a callable object" | use `tilelang.compile(prim, pass_configs=...)` | [0.1.12] |
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
