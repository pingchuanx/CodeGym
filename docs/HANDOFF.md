# CodeGym 工作交接报告

> 编写时间：2026-06-16
> 目的：让接手者能无缝衔接本会话的工作。报告包含已完成内容、进行中任务、关键文件、下一步行动与环境速查。

---

## 0. 重要背景与全局约束（先读这一节）

1. **上下文是重建的**：本会话开始时用户的聊天记录全部丢失，VS Code 调试日志只剩 `session_start` 遥测、无对话内容。当前所有上下文是通过工作区文件 + git + 运行进程**逆向重建**的，原始对话无法恢复。
2. **GPU 纪律（硬约束）**：用户有其它任务要用 GPU。本会话**全程未启动任何 GPU 任务**，所有验证都用 `CUDA_VISIBLE_DEVICES=""` 在 CPU 上跑。接手者在做"真实训练联调"前应先与用户确认 GPU 是否空闲。
3. **训练已结束**：之前的 GRPO 训练（PID 209766）已于 02:19 正常跑完 20/20，GPU 已释放。

---

## 1. 本会话任务总览

| # | 任务 | 状态 |
|---|---|---|
| 1 | 理解训练/数据加载代码与 HF 数据集结构 | ✅ 完成 |
| 2 | 完善"使用完整 HF 数据集"的代码实现（不占 GPU） | ✅ 完成并验证 |
| 3 | 等待 GRPO 训练跑完 | ✅ 完成 |
| 4 | 分析最终实验数据 + 展示完整原始对话记录 | ✅ 完成 |
| 5 | **veRL(expa2026) × CodeGym online rollout 对接** | 🔶 进行中（脚手架已成+CPU验证通过，待数据脚本与真实联调） |

---

## 2. 已完成工作详述

### 2.1 任务 2 — 完整 HF 数据集的训练代码实现

**文件**：[training/train_grpo_codegym.py](../training/train_grpo_codegym.py)（536 → 632 行）

**动机**：旧脚本只能读单个 `example/training_instance.jsonl`、环境只能从磁盘 `example_envs/*.py` 加载。而 HF 完整数据集（`data/codegym_hf/`）是 parquet，每条样本的 `extra_info.gym_env` 字段**自带环境源码**，可内存加载。

**改动（全部向后兼容旧 example 小集）**：
- 新增 `load_env_class_from_source(code, env_name, cache_key)`：从 `gym_env` 源码字符串内存 `exec` 加载环境类（带缓存）。
- 新增 `make_env_from_item(item, env_dir)`：优先用样本自带源码建环境，回退到旧 `.py` 文件。
- 新增 `_iter_raw_records(data_path)`：统一读取 jsonl 文件 / parquet 文件 / **parquet 目录（自动 glob 多 shard）**。
- 新增 `_normalize_prompt(item)`：把 parquet 的 `prompt`（`numpy.ndarray`）规整为 `list[dict]`。
- `load_dataset()` 增加 `max_instances` 参数；新增 CLI `--max_instances`。
- rollout 主循环里 `make_env(...)` 调用改为 `make_env_from_item(...)`。

**验证**（纯 CPU）：`py_compile` 通过；端到端集成测试从 parquet 加载 5 条 → prompt 转 list → 源码内存建环境 → `step()/get_ref_answer()` 正常 → 缓存命中正常。

**⚠️ 数据集下载（`data/` 未随仓库提交，克隆后需自行下载）**：

完整 HF 数据集来自 HuggingFace dataset [`VanishD/CodeGym`](https://huggingface.co/datasets/VanishD/CodeGym)（约 1.3G），已在 `.gitignore` 排除、**不随仓库提交**。克隆仓库后需手动下载到 `data/codegym_hf/`：

```bash
# 方式一（推荐）：huggingface-cli，直接下到 data/codegym_hf/，与下方训练命令路径一致
huggingface-cli download VanishD/CodeGym --repo-type dataset --local-dir data/codegym_hf

# 方式二：仓库自带脚本 utils/model_download_hf.py（已支持 --repo_type dataset）
#   注意它会下到 <local_dir>/<repo_id>，即 data/VanishD/CodeGym/，训练时需把 --data_file 相应调整
python utils/model_download_hf.py --repo_id VanishD/CodeGym --repo_type dataset --local_dir data
```

下载后结构：`data/codegym_hf/{envs_en, envs_cn, task_en_instruction_en_env, task_*_instruction_*_env}/*.parquet`。

**用完整数据集训练的命令**（环境源码内置，无需 `--env_dir`）：
```bash
python training/train_grpo_codegym.py \
  --data_file data/codegym_hf/task_en_instruction_en_env \
  --max_solve_round 12 --max_instances 200 \
  --steps 20 --prompts_per_step 2 --group_size 8
```

### 2.2 任务 4 — 实验数据分析结论

**数据**：[training/output/metrics.jsonl](../training/output/metrics.jsonl)（20 步）、[training/output/rollout_records.jsonl](../training/output/rollout_records.jsonl)（10 条完整轨迹，覆盖 step 1/5/10/15/20）。LoRA adapter 已存于 [training/output/lora_adapter/](../training/output/lora_adapter)。

**结论：训练有效但稀疏二元奖励致波动剧烈。** 解题率（task_reward）分段均值：

| 阶段 | task_R | 格式正确率 | Done率 | 平均轮数 |
|---|---|---|---|---|
| 前段 1–5 | 0.237 | 0.965 | 0.838 | 7.54 |
| 中段 6–12 | 0.286 | 0.861 | 0.697 | 6.30 |
| 后段 13–20 | 0.477 | 0.931 | 0.875 | 6.56 |
| 末 5 步 16–20 | **0.500** | 0.969 | 0.938 | 6.59 |

- 解题率 ~0.24 → ~0.50（翻倍），峰值 step18/19 达 **0.812**。
- 相邻步可从 0.0（step17）跳到 0.812（step18）：G=8 组内若全错则优势为零、无梯度，叠加小 batch（prompts/step=2）放大方差。
- 格式能力很快饱和（fmt 末段 0.97+），瓶颈在"算得对不对"而非"会不会发函数调用"。
- 总耗时 124.5 分钟，平均 374s/step。

---

## 3. 进行中：veRL × CodeGym online rollout 对接（任务 5）

> 这是最需要交接的部分。目标：用 `expa2026` 的 veRL 在 CodeGym 上做 online rollout 的 multi-turn agentic GRPO/PPO 训练。

### 3.1 关键事实
- **`expa2026` = `/home2/zydc/code/expa2026/`**（在本工作区之外，同机可读），内含 **veRL 0.7.0.dev**（vLLM 0.12 + torch 2.9）。
- veRL 对接 HTTP 环境的入口：`verl/earl_http_env/`，模式 = `GenericHttpEnvTool` + 每环境 `adapter` + yaml config。

### 3.2 两边协议差异（= 为什么需要桥接）

| 维度 | CodeGym HTTP 服务 | veRL `GenericHttpEnvTool` 期望 |
|---|---|---|
| 拓扑 | **两级**：manager(:8000) `/get_instance` 分配 worker 端口 → worker 按 `session_id` 操作 | **单级** `base_url` + 单个 `env_id` |
| 创建 | `POST /start {session_id, env_str}`（create+reset 合体） | `POST /create` → 返回 `env_id`，再 `POST /reset` |
| 步进 | `POST /step` → `{status, observation}` | `POST /step` → 从**同一响应**抽 `reward/done/obs` |
| 奖励/终止 | **独立** `GET /reward`、`GET /is_done` | 期望嵌在 step 响应里 |
| 关闭 | `POST /stop` + manager `/release_instance` | `POST /close` |
| 动作 | JSON 串 `{"name","parameters"}` | adapter `build_action(parameters)` 生成 |

### 3.3 采用方案 A：桥接 shim（veRL 零改动）

在 CodeGym 端加一个薄 FastAPI，对外暴露 veRL 期望的 RESTful 协议，内部复用已验证的 `example_rollout_env.codegym_v1`（已封装 manager+session+重试），并在 `/step` 后**回填** `reward/done`。

**已交付文件**：
- [online_server/verl_bridge_server.py](../online_server/verl_bridge_server.py)（245 行）— 桥接服务。端点：`/create /reset /step /close /observation /available_actions /health`。
- [online_server/configs/codegym.yaml](../online_server/configs/codegym.yaml)（62 行）— veRL 工具配置。`env_type: codegym` 未注册 → 自动 fallback 到 `GenericAgentGymAdapter`，故 veRL 无需改动。工具名 `codegym_call`。

**CPU 验证**：用 `fastapi.testclient.TestClient` + mock 假环境跑通完整生命周期（create→step→step(Done)→close）、reward/done 回填、dict 动作序列化、404/400 边界，**14 项全部 PASS**（无需真实 CodeGym 服务、不占 GPU）。

### 3.4 完整数据流（已确认，关键）

veRL 把"每条样本要起哪个 CodeGym 环境"通过数据集逐样本透传，链路为：

```
数据集行 extra_info.tools_kwargs        (utils/dataset/rl_dataset.py:349,355)
   → non_tensor_batch 逐列              (experimental/agent_loop/agent_loop.py:656)
   → agent loop kwargs["tools_kwargs"]
   → tools_kwargs[tool_name]["create_kwargs"]   (earl_tool_agent_loop.py:892)
   → tool.create(create_kwargs=...)             (earl_tool_agent_loop.py:901)
   → shim /create 读 create_payload.env_str     (verl_bridge_server.py:_resolve_env_str)
   → codegym_v1.from_env_str(env_str)
```

因此 **veRL parquet 每行 `extra_info` 需形如**（`codegym_call` 必须与 yaml 的 `tool_name` 一致）：
```json
{
  "extra_info": {
    "index": 0,
    "need_tools_kwargs": true,
    "tools_kwargs": {
      "codegym_call": {
        "create_kwargs": {
          "create_payload": { "env_str": "codegym_v1@<code_id>__<EnvName>@{json任务参数}" }
        }
      }
    }
  }
}
```
> `env_str` 就是现有数据集里的 `ability` 串，可直接复用。

### 3.5 任务 5 剩余待做（按优先级）

1. **数据转换脚本（纯 CPU，可立即做）**：把 `example/training_instance.jsonl` 或 `data/codegym_hf/task_*` 转成 veRL 训练用 parquet——核心是按 3.4 把每行 `ability` 写进 `extra_info.tools_kwargs.codegym_call.create_kwargs.create_payload.env_str`，并设 `need_tools_kwargs=true`。保留 `prompt/reward_model/data_source`。
2. **真实 shim↔CodeGym 联调（基本不占 GPU）**：
   - 启动 manager：`python online_server/online_server/env_instance_manager.py --workers N`（需先按 README 把 HF 环境放到 `online_server/online_server/envs/codegym_v1/`）。
   - 启动 shim：`cd online_server && SERVER_IP_ADDRESS=:: BRIDGE_PORT=8200 python verl_bridge_server.py`。
   - 用 `curl` 走一遍 `/create → /step → /close`，确认真实环境的 reward/done 回填正确。
   - ⚠️ 注意：CodeGym worker 跑的是**纯 Python gym 环境**（CPU），但确认机器 GPU 占用情况后再大规模铺 worker。
3. **veRL 训练联调（需 GPU，先与用户确认）**：在 veRL 侧把 `configs/codegym.yaml` 的 `base_url` 指到 shim，用 `earl_tool_agent_loop` + `main_ppo.py`（`algorithm.adv_estimator=grpo`、`rollout.multi_turn.enable=true`）跑通最小规模。

### 3.6 已知风险/待确认点
- **动作格式对齐**：generic adapter 暴露 `codegym_call(action="<JSON字符串>")`，模型需在 `action` 字段里放 CodeGym 的 `{"name","parameters"}` 串。若模型不照做，可能需要写一个 CodeGym 专用 adapter（方案 B 的轻量版）替代 generic。这是端到端联调时最可能踩的坑。
- **`reset_payload` 噪声**：`GenericHttpEnvTool.create` 会默认塞 `game/world_type` 等 alfworld 残留字段进 reset_payload；shim 的 `/reset` 忽略 body 只回缓存 obs，无害，但联调时注意日志。
- **observation/available_actions**：CodeGym 不枚举可用动作，shim 的 `/available_actions` 返回空列表，yaml 已设 `refresh_available_actions: false`。

---

## 4. 关键文件清单

| 文件 | 说明 |
|---|---|
| [training/train_grpo_codegym.py](../training/train_grpo_codegym.py) | 单文件 GRPO 训练（本会话增强为支持完整 HF parquet 数据集） |
| [training/output/metrics.jsonl](../training/output/metrics.jsonl) | 20 步训练指标 |
| [training/output/rollout_records.jsonl](../training/output/rollout_records.jsonl) | 10 条完整原始对话轨迹（机读） |
| [training/output/rollout_records.md](../training/output/rollout_records.md) | 同上人读版（每轮截断 160 字符） |
| [training/output/lora_adapter/](../training/output/lora_adapter) | 训练产出的 LoRA adapter |
| [online_server/verl_bridge_server.py](../online_server/verl_bridge_server.py) | **新增** veRL↔CodeGym 桥接 shim |
| [online_server/configs/codegym.yaml](../online_server/configs/codegym.yaml) | **新增** veRL 工具配置 |
| [online_server/example_rollout_env.py](../online_server/example_rollout_env.py) | CodeGym 在线环境客户端（shim 复用其 `codegym_v1`） |
| [docs/CODEGYM_DEEP_DIVE.md](CODEGYM_DEEP_DIVE.md) | 项目深入文档（第 7 章讲 veRL 对接思路，第 1210 行提到 expa2026 路径） |

**workspace 外（同机）**：
- `/home2/zydc/code/expa2026/verl/earl_http_env/` — `generic_env_tool.py` / `client.py` / `adapters/` / `registry.py`
- `/home2/zydc/code/expa2026/verl/experimental/agent_loop/earl_tool_agent_loop.py` — EARL agent loop
- `/home2/zydc/code/expa2026/verl/utils/dataset/rl_dataset.py` — 数据集（`tools_kwargs` 透传，行 349/355）

---

## 5. 环境与命令速查

| 用途 | 环境 | 说明 |
|---|---|---|
| GRPO 训练 / 跑 gym 环境 | conda `verl070`（`/home2/zydc/anaconda3/envs/verl070/bin/python`） | 有 gymnasium/torch/vLLM/fastapi/httpx；base 环境**无** gymnasium |
| CodeGym 在线服务 | conda `online_server` | 有 fastapi/uvicorn/requests，**无** httpx |
| CPU 验证桥接 | `verl070` | 同时有 fastapi+TestClient+httpx，单环境可验证 |

- **CPU 验证铁律**：所有验证命令前加 `CUDA_VISIBLE_DEVICES=""`，确保不占 GPU。
- **终端注意**：本机终端工具会把开头的 `cd A && ...` 简化掉导致 cwd 漂移；对 workspace 外文件用绝对路径，或确保命令第一段就是 `cd`。

---

## 6. 接手者下一步建议（不占 GPU 即可推进）

1. 写 3.5(1) 的**数据转换脚本**，产出一个 veRL parquet 样例（哪怕几十条），人工核对 `extra_info.tools_kwargs` 结构。
2. 做 3.5(2) 的**真实 shim↔CodeGym 联调**（manager+shim+curl），验证 reward/done 回填。
3. 在小样本上把动作格式对齐问题（3.6 第一条）摸清，决定是否需要 CodeGym 专用 adapter。
4. 以上无误后，再与用户确认 GPU 档期，做 veRL 端到端训练联调。
