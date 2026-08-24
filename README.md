# TileOPs Foundry Loop

这个仓库只负责把一个 TileOPs round 准备好，并把 Agent 的工作环境放进
TileOPs runner Docker。Foreman 仍然创建 worktree、space 和 pane；pane 里运行
Pi，Pi 的文件和命令工具通过 SSH 转发到 round 容器。

```text
new_round.py
    -> copy templates/round -> rounds/<slug>
dispatch_round.sh
    -> build TileFoundry wheel
    -> write worker admission
    -> foreman assign solo (kind=pi)
post-worktree.sh
    -> build SSH-enabled runner image
    -> start one container
worker-env.sh
    -> export Pi SSH target
    -> pane starts scripts/pi
```

## 目录边界

| 位置 | 用途 |
| --- | --- |
| `templates/round/` | 每轮工作目录的唯一模板，包含说明、knowledge 和证据目录 |
| `rounds/<slug>/` | 一轮实际工作目录；Agent 的 HIR、脚本、日志和报告都放这里 |
| TileOPs worktree | Agent 修改的生产 kernel/dispatch 目标 |
| Docker | TileLang、CUDA、TileFoundry wheel 和 GPU 运行时 |
| 宿主机 | Git commit、push、PR、CI 和最终审阅 |

容器只挂载当前 round 和当前 TileOPs worktree。loop 仓库、历史 round、宿主
TileFoundry checkout 和 Git common dir 都不需要挂载。

## 配置

```bash
cp config/local.env.example .env
# 修改路径和 GPU；Docker daemon 使用当前机器的配置
source .env
```

需要本机已经安装并配置：`docker`、`foreman`、`herdr` 和 `pi`。Pi 的 provider
认证继续使用宿主机配置；只有 Pi 的工作工具通过 SSH 进入容器。

loop 的 dispatch 会显式传 `foreman assign ... --kind pi`。这不会改动 Foreman
的默认配置；`[modes.solo.agent]` 仍然可以继续使用 Claude。机器上的
`foreman.toml` 只需要为显式的 `pi` kind 提供命令映射。

每轮可以选择 Pi 使用的 provider/model：

```bash
TILEOPS_ROUND_MODEL=anthropic/<claude-model-id> \
TILEOPS_ROUND_EFFORT=xhigh ./scripts/dispatch_round.sh <task> <branch> <brief>

TILEOPS_ROUND_MODEL=openai-codex/<gpt-model-id> \
TILEOPS_ROUND_EFFORT=xhigh ./scripts/dispatch_round.sh <task> <branch> <brief>
```

不设置 `TILEOPS_ROUND_MODEL` 时，Pi 使用自己保存的 provider/model。

## 使用

```bash
python scripts/new_round.py \
  --slug fused-moe-r1 \
  --operator fused_moe \
  --scope "fused MoE expert projection" \
  --baseline "vLLM" \
  --tileops-repo "$TILEOPS_REPO" \
  --tilefoundry-repo "$TILEFOUNDRY_REPO" \
  --root "$TILEOPS_LOOP_STATE_ROOT"

./scripts/dispatch_round.sh \
  fused-moe-r1 perf/fused-moe-r1 \
  "$TILEOPS_LOOP_STATE_ROOT/fused-moe-r1/brief.md"
```

round 创建后，先看 `rounds/<slug>/brief.md`。Agent 会在同一个目录中留下
证据；容器销毁不会删除这些文件。工作完成后，回到宿主 worktree 执行项目自身
的测试、commit、push 和 PR 命令。loop 不替宿主项目定义 PR 格式。

结束一轮：

```bash
./scripts/stop_round.sh <task> <round-dir> <worktree>
```

`stop_round.sh` 只停止容器并调用 `foreman done`；round 目录和 worktree 默认保留。
