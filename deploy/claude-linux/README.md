# DeepTutor（Claude 订阅版）Linux 部署手册

目标：服务、模型调用和本地模型都跑在 Linux 主机上，Mac 只开浏览器，经 Tailscale 访问 `https://<host>.<tailnet>.ts.net`。

下文标「Linux」的命令在主机上执行（从 Mac `ssh <linux-host>` 进去即可）；标「Mac」的只在 Mac 上执行，因为只有 Mac 能从 1Password 取 token。

## 组成

| 容器 | 镜像 | 作用 |
|---|---|---|
| `deeptutor` | `deeptutor-claude:local`（本目录的 `Dockerfile`） | 上游 1.6.15 镜像 + 本 fork 的后端 + Claude Agent SDK。只在主机的 `127.0.0.1:3782` 发布前端；前端在容器内把 `/api/*`、`/ws/*` 转给 8001 上的后端，所以 8001 不发布 |
| `deeptutor-sandbox-runner` | `deeptutor-runner:local`（仓库根目录的 `Dockerfile.runner`） | 执行模型生成的代码。只读挂载工作区，只能写 `outputs/`，看不到 `data/system` |
| `deeptutor-ollama` | `ollama/ollama:0.40.12` | embedding：`qwen3-embedding:0.6b`，1024 维 |
| `deeptutor-speaches` | `ghcr.io/speaches-ai/speaches:0.9.0-rc.3-cpu` | 语音，OpenAI 兼容接口：Kokoro 朗读（`/v1/audio/speech`），faster-whisper 听写（`/v1/audio/transcriptions`） |

- 数据都在仓库根目录的 `data/`（已被 git 忽略），也就是 `~/dev/DeepTutor/data`，备份这一个目录即可。模型在命名卷 `deeptutor_ollama`、`deeptutor_speaches-hf` 里。
- runner 只接在 `deeptutor-network` 上，ollama 和 speaches 只接在 `models` 上，两个网络都接的只有 `deeptutor`。runner 里的代码因此连不到本地模型服务。
- 视频生成不配置。没有视频模型时，DeepTutor 每轮会自动去掉这个工具。

| 文件 | 用途 |
|---|---|
| `Dockerfile` | overlay 镜像。只用 legacy builder 能构建的语法，因为主机没装 buildx |
| `compose.yaml` | 四个服务，项目名 `deeptutor` |
| `compose.gpu.yaml` | GPU 覆盖：ollama、speaches 申请 NVIDIA GPU，speaches 换成 CUDA 镜像 |
| `up.sh` | 从 stdin 读 token，带着它执行 `docker compose up -d --remove-orphans` |
| `seed/` | 首次启动后的配置。`seed.sh` 先下载模型，再用 DeepTutor 自己的 CLI（`setup.json`）和设置服务（`catalog.py`）写配置；`test_catalog.py` 是 `catalog.py` 的单测，在仓库根目录用 `python -m pytest -q deploy/claude-linux/seed/test_catalog.py` 运行 |

## 前提

Linux 主机：

- Docker 和 Compose v2。不需要 buildx：两个镜像都用 `docker build`（legacy builder）构建。
- git。
- Tailscale 已登录，并已执行过 `sudo tailscale set --operator=$USER`，之后 `tailscale serve` 不用 sudo。
- tailnet 后台开了 MagicDNS 和 HTTPS 证书。没开的话，第一次 `tailscale serve` 会给出开启链接。
- 内存：overlay 构建不跑前端的 `next build`，空闲 2.5 GB 左右就能构建。运行时的内存上限写在 `compose.yaml` 里，是上限不是预留；ollama 和 speaches 空闲 5 分钟后会卸载模型。

Mac：

- 1Password CLI `op`，并且能 `ssh <linux-host>`。
- 在任意装了 `claude` 的电脑上运行 `claude setup-token`，把生成的 token 存进 1Password（有效期一年）。不要换成 API key：容器里的 `ANTHROPIC_API_KEY` 是故意置空的。

## 首次部署

### 1. 拉代码（Linux）

```bash
git clone https://github.com/<owner>/DeepTutor.git ~/dev/DeepTutor
```

### 2. 构建两个镜像（Linux）

```bash
cd ~/dev/DeepTutor
docker build -f deploy/claude-linux/Dockerfile -t deeptutor-claude:local .
docker build -f Dockerfile.runner -t deeptutor-runner:local .
```

第一条会拉 `ghcr.io/hkuds/deeptutor:1.6.15`（压缩后约 818 MB），再装 `claude-agent-sdk==0.2.174`，它自带约 290 MB 的 Claude CLI。没装 buildx 时 `docker build` 会提示 legacy builder 已弃用，可以忽略。

### 3. 启动（Mac）

先在 Mac 上建一个只有引用、没有真值的模板，放在仓库外面，比如 `~/.config/op/deeptutor-linux.env.tpl`：

```
CLAUDE_CODE_OAUTH_TOKEN=op://<vault>/<item>/<field>
```

首次部署和以后每次重新部署，都在 Mac 终端里跑这一条：

```bash
op run --env-file="$HOME/.config/op/deeptutor-linux.env.tpl" -- sh -c 'printf "%s\n" "$CLAUDE_CODE_OAUTH_TOKEN" | ssh <linux-host> "cd ~/dev/DeepTutor && deploy/claude-linux/up.sh"'
```

token 的去向：

- `op run` 只把它注入这一条命令的环境变量。`printf` 是 shell 内建命令，token 不会出现在任何进程的命令行参数里，经 ssh 的加密通道写进 `up.sh` 的 stdin。
- `up.sh` 用 `IFS= read -r` 读取：读到空值会拒绝，stdin 是终端也会拒绝。它不回显、不写文件，只 export 给 `docker compose`。
- 此后 token 只存在于 Docker 保存的容器配置里（`/var/lib/docker/containers/<id>/config.v2.json`，只有 root 能读；docker 组的成员能用 `docker inspect` 看到）。
- 主机重启不受影响：`restart: unless-stopped` 的容器由 dockerd 按保存的配置拉起，token 还在。
- 在主机上直接跑 `docker compose up` 会失败，报 `required variable CLAUDE_CODE_OAUTH_TOKEN is missing a value`。这是故意的，防止起一个没有 token 的容器。`ps`、`logs` 等其他 compose 子命令同样会失败，平时用 `docker logs|exec|restart <容器名>`。
- 换 token：把新值存进 1Password，再跑一遍上面的命令，compose 会用新值重建 `deeptutor` 容器。

### 4. 下载模型并写配置（Linux）

```bash
~/dev/DeepTutor/deploy/claude-linux/seed/seed.sh
```

可以重复运行，结果不变；但它管的几项（active 模型、语音模型、登录开关）以脚本为准，重跑会覆盖你在界面上对这几项的改动。它依次做这些事：

1. 等 `deeptutor`、`deeptutor-ollama`、`deeptutor-speaches` 都变成 healthy。
2. `ollama pull qwen3-embedding:0.6b`（约 640 MB）；让 speaches 下载 `speaches-ai/Kokoro-82M-v1.0-ONNX`（约 350 MB）和 `Systran/faster-whisper-small`（约 490 MB）。speaches 收到请求时不会自动下载模型，所以要先下好，模型存在卷里。
3. `python -m deeptutor_cli config apply`，用的是 `setup.json`：LLM 用 `claude_agent_sdk` 并设为 active；embedding 用 Ollama，地址 `http://ollama:11434/api/embed`，1024 维；搜索用 DuckDuckGo；端口固定为 8001/3782。
4. `catalog.py` 通过 DeepTutor 的设置服务，补上 `config apply` 写不了的部分：
   - Claude 模型 `claude-sonnet-5-5`（active）、`claude-opus-5-5`、`claude-haiku-5-5`、`claude-fable-5-1`。每个都写 1M 上下文（来源 manual），开 tools 和 vision、关 json_output，推理档位 low 到 max。
   - 朗读：binding 用「OpenAI Compatible」（`custom`），地址 `http://speaches:8000/v1`。两个模型：「Kokoro zh」（音色 `zf_xiaobei`，active）和「Kokoro en」（音色 `af_heart`）。speaches 按音色决定 Kokoro 的语言，中英文在「设置 → 语音」里切换。CPU 上合成长回复较慢，超时设为 300 秒。
   - 听写：同一个地址，模型 `Systran/faster-whisper-small`，语言留空表示自动识别，适合中英混说。
   - 生图：Codex 适配器，binding `openai_codex`，模型 `gpt-image-2`。要等第 7 步登录 Codex 之后才能用。
   - `auth.json` 打开登录；`system.json` 设 `sandbox_allow_subprocess=false`，代码只在 runner 里执行。这两个文件只在启动时读取，有改动时脚本会自动 `docker restart deeptutor`。

### 5. 开放给 tailnet（Linux，只做一次）

```bash
tailscale serve status    # 先确认 443 上没有别的配置，下一条会覆盖它
tailscale serve --bg 3782
```

之后在 Mac 上打开 `https://<host>.<tailnet>.ts.net`。`--bg` 的配置由 tailscaled 保存，重启后仍然有效；要关掉用 `tailscale serve --https=443 off`。不要用 `tailscale funnel`，那会开放到公网。serve 用 Go 的反向代理转发，应当能通过 WebSocket，首次部署后在聊天里确认一下流式回复。

### 6. 注册管理员（浏览器）

先确认第 4 步已经跑完，并且 `docker ps` 里 `deeptutor` 重启后又变回 healthy（登录要在这次重启后才生效），再打开 `https://<host>.<tailnet>.ts.net/register`。第一个注册的账号自动成为 admin，之后注册自动关闭。不要再加别的账号。

### 7. 登录 Codex（浏览器，生图要用）

DeepTutor 在远程主机上，授权后的回调地址 `http://localhost:1455/auth/callback` 在 Mac 上打不开，所以用粘贴回调地址的方式：

1. Claude 必须已经是 active LLM（第 4 步已设好）。当时如果没有 active LLM，Codex 登录成功后会被自动设为 active。
2. 在设置里找到 OpenAI Codex，点「使用 Codex 登录」，再点「打开授权页面」，在 ChatGPT 页面登录并同意。
3. 浏览器跳到 `http://localhost:1455/auth/callback?code=…&state=…` 后打不开，这是正常的。把地址栏里的完整地址粘到「回调地址」，点「完成登录」。
4. 整个过程要在 300 秒内完成，超时就重来。

界面在远程访问时还会显示一条 SSH 隧道命令，用不到。仓库根目录的 `compose.codex-oauth.yaml` 只适用于浏览器就在 Docker 主机上的情况，这里也用不到。

### 8. 检查（Linux）

```bash
docker ps --format '{{.Names}}\t{{.Status}}'      # 四个容器都应是 healthy
docker exec deeptutor sh -c 'test -n "$CLAUDE_CODE_OAUTH_TOKEN" && echo "token: set"'
docker exec -u deeptutor deeptutor sh -c 'command -v claude || echo "claude: not on PATH"'
docker exec -u deeptutor -w /app deeptutor python -m deeptutor_cli doctor --format json
```

然后在网页里发一条消息，确认回复是流式出来的，这同时验证了订阅可用。不要用 `doctor --online` 来验证：登录开着时，`docker exec` 起的 CLI 进程没有用户身份，owner 守卫会直接拒绝，不发请求也不计费，doctor 报 Provider response FAIL 并以 1 退出，看上去像 token 坏了，其实不是。要单独测一次模型，就用 owner 账号在「设置」里对 Claude 模型点测试，测试会带上当前登录的用户。

最后在「设置 → 语音」里各测一次朗读和听写。

## 日常维护

- **重新部署**：在 Mac 上跑第 3 步那条命令。
- **更新代码**（Linux）：

  ```bash
  cd ~/dev/DeepTutor && git pull --ff-only
  docker build -f deploy/claude-linux/Dockerfile -t deeptutor-claude:local .
  docker build -f Dockerfile.runner -t deeptutor-runner:local .
  ```

  然后在 Mac 上跑第 3 步；`seed/` 有变化的话再跑第 4 步。compose 发现镜像变了，会重建对应的容器。fork 同步了上游新版本时，把 `Dockerfile` 的 `FROM` 标签改成同一个版本：overlay 只替换后端代码，不会重装依赖。
- **看日志**（不会带出 token）：`docker logs -f --tail 200 deeptutor`，其他容器同理。不要用 `docker compose logs`（缺 token 会失败），也不要用 `docker inspect deeptutor` 或 `docker compose config` 看配置，这两条会把 token 打印出来。
- **看内存**：`docker stats --no-stream`。
- **重启**：`docker restart deeptutor`，token 保留。不要 `docker compose down`：容器删了，就得从 Mac 重新部署。

### 切换到 GPU

1. 在主机上由你本人用 sudo 执行：按 NVIDIA 官方文档安装 `nvidia-container-toolkit`，然后 `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。重启 docker 会连带重启主机上的所有容器，重启策略是 `no` 的容器要手动 `docker start`。
2. `nvidia-smi` 显示的 CUDA Version 要不低于 12.6，因为 GPU 版 speaches 用的是 `0.9.0-rc.3-cuda-12.6.3`。
3. 在 Mac 上再跑一次第 3 步。`up.sh` 在 `docker info` 里看到 nvidia runtime，就会叠加 `compose.gpu.yaml`，重建 ollama 和 speaches；模型在卷里，不用重新下载。
4. 检查：做一次 embedding（比如往知识库加个小文件）之后，`docker exec deeptutor-ollama ollama ps` 的 PROCESSOR 列应显示 GPU。
5. 有了 GPU，可以把听写换成更大的 faster-whisper 模型：先 `docker exec deeptutor-speaches curl -fsS -X POST http://localhost:8000/v1/models/<模型 id>` 下载，再在「设置 → 语音」里改。要长期使用，就同时改 `seed/catalog.py` 的 `WHISPER` 和 `seed/seed.sh` 的下载列表，否则重跑 seed 会改回 small。

## 安全说明

- **token**：只在 `deeptutor` 容器的环境变量里，以及 Docker 保存的容器配置里。DeepTutor 的代码不读它，SDK 把环境变量原样传给内置的 Claude CLI。`ANTHROPIC_API_KEY` 和 `ANTHROPIC_AUTH_TOKEN` 被置空，因为它们在 CLI 里的优先级比订阅 token 高，一旦有值，计费就会变成按 API 收费。
- **信任边界是整个 `deeptutor` 容器**。同容器、同 uid 的进程都能从 `/proc/<pid>/environ` 读到 token，前端的 Node 进程也继承了它。所以：
  - 不要设置 `DEEPTUTOR_EXTRAS`、`DEEPTUTOR_APT_PACKAGES`，不要装 `math-animator` extra（它执行模型生成的代码时不经过沙箱）；
  - 镜像里不装任何 agent CLI，`claude` 不在 PATH 上，只有 SDK 内置的那一份；
  - 子代理接口 `/api/subagents` 只对 admin 开放（fork 的改动，见 `deeptutor/api/routers/subagents.py`）；
  - 模型生成的代码只在 runner 里执行。seed 把 `sandbox_allow_subprocess` 设成了 false，runner 挂掉时也不会退回主容器里执行；
  - 管理界面里有三处会在这个容器里、以同一个 uid 跑第三方代码，它们同样读得到 token：stdio 方式的 MCP 服务器、CLI 应用安装（会执行包里的 `setup.py`）、解析引擎的一键安装（`pip install`）。MCP 尽量用 HTTP/SSE 方式接入，CLI 应用和解析引擎只装信得过的。
- **runner**：只读挂载工作区，只能写 `outputs/`，没有 `data/system`，访问不到 ollama 和 speaches。但它和 DeepTutor 在同一个网络里，所以登录必须一直开着。
- **端口**：只发布 `127.0.0.1:3782`，tailnet 访问走 `tailscale serve` 的 HTTPS。Docker 发布的端口会绕过 ufw，不要把它改成所有网卡。
- **账号**：只保留你一个账号。partner（IM 机器人）只有在你本人聊天时才能用 Claude 订阅，别人发起的回合会被拒绝；但它生图用的是 owner 的 Codex 登录，所以不要把 partner 开放给别人。
- `auth.json` 的 `cookie_secure` 保持默认的 false：前端和 API 同源，SameSite=Lax 就够了；设成 true 反而会把 cookie 改成 SameSite=None。

## 已知限制

- 主机没有 buildx，`docker build` 走 legacy builder，会把整个仓库目录（扣除 `.dockerignore` 排除的部分）发给本机的 dockerd，其中包括 `data/` 下没被排除的子目录，比如 `data/system`。这些文件只在本机内传递，不会进镜像（`Dockerfile` 只 COPY 两个源码目录和 `pyproject.toml`），但 `data/` 很大时构建会变慢。
- 前端沿用上游镜像里构建好的版本，fork 在 `web/` 下的改动不会生效。目前只有一处：Claude Agent 在设置页的图标，界面上会显示成默认图标。
- speaches 固定在 `0.9.0-rc.3`：这是目前最新的带版本号构建，GitHub release 没有标成 prerelease；`latest-cpu` 仍指向 0.8.3。
