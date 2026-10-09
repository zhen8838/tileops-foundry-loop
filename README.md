# TileOPs Foundry Loop

这个仓库只为 Foreman 提供 TileOPs round 环境。下载后运行一次 `./setup`；以后直接
使用 `foreman assign`，不再经过 loop 命令。

```text
首次：./setup
  ├─ 检查 Foreman、herdr、Claude Code 等本机命令
  ├─ 构建含 SSH 和 TileOPs runtime 的 Docker image
  └─ 将 hook/template 软链接到 foreman/local

每轮：foreman assign ... --kind claude
  ├─ Foreman 创建 TileOPs worktree 和 pane
  ├─ hook 复制 brief、knowledge 和 PR gate -> rounds/<task>
  ├─ hook 起该轮容器：TileFoundry 源码 editable 安装，Claude Code 连凭据一起挂进去
  ├─ hook 在 round 放一个 claude 垫片并加进 PATH
  └─ herdr 起的 agent 经垫片 ssh 进容器，在 /workspace/round 工作

结束：foreman done <task> --rm
  ├─ pre_done hook 导出 foundry.patch，并归档容器内的 transcript
  ├─ pre_done hook 删除本轮容器、SSH alias 和临时 key
  └─ Foreman 删除 pane/worktree，保留 round 目录
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `setup` | 唯一安装入口 |
| `container/` | Agent image |
| `integrations/` | Foreman hook 和固定的起始 prompt |
| `template/` | 每轮复制的 brief、knowledge、work 和 evidence |
| `rounds/<task>/` | 该轮实验、证据、环境记录、报告和 session transcript |

容器只挂载当前 round、当前 TileOPs worktree、该 worktree 的 Git common dir、
该轮的 TileFoundry 源码 worktree 和 runtime cache。loop 仓库和其他 round 不挂载。Git、GitHub 和 SSH 认证以只读方式从宿主提供给容器，Agent 可以在
`/workspace/tileops` 按项目规则完成 commit、push 和 PR；PR 不在 loop 内处理。

## 安装

```bash
cp .env.example .env
# 修改 TILEOPS_REPO、TILEFOUNDRY_REPO 和可选的机器配置
./setup
```

需要本机已有 `claude`、`docker`、`foreman`、`gh`、`herdr`、`uv`、`git`、`ssh` 和
`nvidia-smi`。Agent 用的是宿主那份 Claude Code 和 `~/.claude/.credentials.json`，
容器内不另装，也不改 Foreman 的默认配置。

## 启动

```bash
foreman assign solo \
  --project tileops \
  --task fused-moe-r1 \
  --branch feat/moe/fused-moe-r1 \
  --prompt "按 #2270 给 FusedMoEFwdOp 加 swigluoai 激活；只做这一个 op。" \
  --kind claude \
  --effort high
```

`--prompt` 只写一句：引用 issue 号，并说明这一轮做 issue 里的哪一部分。它填进 brief 的
`{{PROMPT}}`；目标、数据、验收线和落点留在 issue 里，brief 会让 Agent 先读完 issue 和
评论，不在 prompt 里再写一份。

模型由 round 目录的 `.claude/settings.json` 钉为 `claude-opus-5-5`，只作用于这一轮的
会话；要换就在命令里加 `--model`，它优先于这份设置。

hook 会先生成 `rounds/fused-moe-r1/brief.md`，再启动 Agent。Foreman 只给 Agent 一句
固定指令：读取当前目录的 `brief.md`；任务、环境、约束和交付物全部在 brief 内。
`check_round.py` 只检查 artifact、命令记录、diff 边界和 PR 内容一致性；它不猜优化
语义。`open_pr.sh` 在 push 前要求 Agent 重新阅读最终 HIR、analyze/measure 证据和
production diff；未完成的 `tilefoundry schedule` 只记录 finding，不阻断 round。

失败会按阶段返回，例如：

```text
tileops setup-worktree: [TileFoundry wheel] failed (exit 1)
  task=fused-moe-r1
  worktree=/home/.../TileOPs-worktrees/tileops-fused-moe-r1
```
