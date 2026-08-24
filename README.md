# TileOPs Foundry Loop

这个仓库只为 Foreman 提供 TileOPs round 环境。下载后运行一次 `./setup`；以后直接
使用 `foreman assign`，不再经过 loop 命令。

```text
首次：./setup
  ├─ 检查 Foreman、Pi 和 Pi 自带的 examples/extensions/ssh.ts
  ├─ 构建含 SSH 和 TileOPs runtime 的 Docker image
  └─ 将 hook/template 软链接到 foreman/local

每轮：foreman assign ... --kind pi
  ├─ Foreman 创建 TileOPs worktree 和 pane
  ├─ hook 复制 brief、knowledge 和 PR gate -> rounds/<task>
  ├─ hook 构建当前 TileFoundry wheel并启动该轮容器
  └─ Pi 通过自带 ssh.ts 在 /workspace/round 工作
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `setup` | 唯一安装入口 |
| `container/` | Agent image |
| `integrations/` | Foreman hook 和固定的起始 prompt |
| `template/` | 每轮复制的 brief、knowledge、work 和 evidence |
| `rounds/<task>/` | 该轮实验、证据、环境记录和报告 |

容器只挂载当前 round、当前 TileOPs worktree、该 worktree 的 Git common dir、
TileFoundry wheel 和 runtime cache。宿主 TileFoundry checkout、loop 仓库和其他
round 不挂载。Git、GitHub 和 SSH 认证以只读方式从宿主提供给容器，Agent 可以在
`/workspace/tileops` 按项目规则完成 commit、push 和 PR；PR 不在 loop 内处理。

## 安装

```bash
cp .env.example .env
# 修改 TILEOPS_REPO、TILEFOUNDRY_REPO 和可选的机器配置
./setup
```

需要本机已有 `docker`、`foreman`、`gh`、`herdr`、`pi`、`uv`、`git`、`ssh` 和
`nvidia-smi`。`setup` 会验证 Pi 自带的 `examples/extensions/ssh.ts` 能提供
`--ssh`；不会安装或维护另一个 SSH 扩展，也不会修改 Foreman 的默认 Claude/Codex
配置。

## 启动

```bash
foreman assign solo \
  --project tileops \
  --task fused-moe-r1 \
  --branch perf/fused-moe-r1 \
  --prompt "用 TileFoundry 优化 fused MoE，并完成验证与 PR" \
  --kind pi \
  --model openai/gpt-5.6-sol \
  --effort high
```

hook 会先生成 `rounds/fused-moe-r1/brief.md`，再启动 Agent。Foreman 只给 Pi 一句
固定指令：读取当前目录的 `brief.md`；任务、环境、约束和交付物全部在 brief 内。
`check_round.py` 拒绝 naive/unplaced HIR、单一 placement、config-only kernel diff 和
不一致的 PR Description；Agent 只能在 gate PASS 后通过 `open_pr.sh` 创建或更新 PR。

失败会按阶段返回，例如：

```text
tileops setup-worktree: [TileFoundry wheel] failed (exit 1)
  task=fused-moe-r1
  worktree=/home/.../TileOPs-worktrees/tileops-fused-moe-r1
```
