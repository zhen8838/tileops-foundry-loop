# TileFoundry: surfaces known to refuse a legal program

Reproduced against an admitted wheel, so a round spends its time on the operator
instead of rediscovering them. Hitting one is still a finding: cite the id, keep
the faithful program, and record the surface as blocked. Never reshape the HIR
into something a surface accepts but the kernel does not do. Promotion here is a
human step, same as `tilelang.md`.

| surface | what it refuses | seen on |
|---|---|---|
| `analyze --timeline`, `schedule --topology` | values in `rmem`/`smem` — *"traffic is only at unmodelled storage level(s) 'rmem'"*, *"must reside in GMEM"*; the target states register and shared bandwidth as unavailable. The provenance gate requires that same local tier, so the two asks are incompatible; `--compute-cost`, `--memory`, `--roofline` accept it | dev37+g0559a4b75 · `TF-LOCAL-STORAGE-UNMODELLED` |
| `analyze --timeline` | any comparison — *"no per-unit compute rate for dtype 'bool'"* — while `--compute-cost` counts that comparison's flops. Every masked attention has this shape | same · `TF-TIMELINE-BOOL` |
| `Mesh` over two topology levels | *"one mesh names multiple topology levels"*; nesting two single-level meshes makes the outer coordinates run-time Exprs. State cta placement only, and leave the lane structure to the kernel source | same · `TF-MESH-LEVELS` |
| `@module` | a refused class takes every other selector in its file with it, because parsing happens at decoration. One reproducer form per file, or put them behind an env switch | same · `TF-MESH-LEVELS` |
