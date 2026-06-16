#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeGym 多轮工具调用 GRPO 训练脚本 (单卡 / LoRA / bf16)
=========================================================

忠实复现 CodeGym 论文的训练设定，在单张 GPU 上训练 Qwen2.5-3B-Instruct：

  * 任务来源 : example/training_instance.jsonl（system=工具文档, user=任务说明）
  * 环境     : 进程内直接驱动 example/example_envs/*.py 的 gymnasium 环境
  * 交互     : 模型多轮输出 function-call（<|FunctionCallBegin|>[{...}]<|FunctionCallEnd|>），
               解析后 env.step() → observation，循环直到 Done 或 max_rounds
  * 奖励     : 回合结束读 env.reward（0/1 二元，仅终局结算，无中间奖励）
  * 算法     : GRPO —— 同一任务采样 G 条轨迹，组内归一化 advantage，
               对 assistant 生成 token 做 token-level PPO-clip policy gradient
  * 显存     : LoRA（peft）+ bf16，单卡即可训练 3B

用法：
    CUDA_VISIBLE_DEVICES=2 python training/train_grpo_codegym.py            # 正式配置
    CUDA_VISIBLE_DEVICES=2 python training/train_grpo_codegym.py --smoke    # 冒烟小配置
"""
import os

# 离线加载本地已缓存的模型权重，避免联网
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 缓解多轮长序列下的显存碎片（OOM 提示明确建议）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import json
import glob
import time
import random
import argparse
import importlib.util
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------
# 1. 数据 & 环境加载
# ----------------------------------------------------------------------
_ENV_CLASS_CACHE = {}
_ENV_SRC_CACHE = {}


def load_env_class(env_dir, mid):
    """从 example_envs/<mid>.py 动态加载环境类。mid = '<code_id>__<EnvName>'。"""
    if mid in _ENV_CLASS_CACHE:
        return _ENV_CLASS_CACHE[mid]
    path = os.path.join(env_dir, mid + ".py")
    spec = importlib.util.spec_from_file_location(f"cgenv_{mid}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env_name = "__".join(mid.split("__")[1:])  # 去掉 code_id 前缀
    cls = getattr(mod, env_name)
    _ENV_CLASS_CACHE[mid] = (cls, env_name)
    return cls, env_name


def load_env_class_from_source(code, env_name, cache_key):
    """从环境源码字符串动态加载环境类并缓存（HF 完整数据集 extra_info['gym_env']）。

    完整数据集每条样本自带环境源码，无需磁盘 .py 文件；cache_key 用 mid
    （'<code_id>__<EnvName>'）以避免不同 code_id 下同名 env 互相覆盖。
    源码来自本项目生成的可信数据集，信任级别与旧版 exec_module 加载 .py 一致。
    """
    if cache_key in _ENV_SRC_CACHE:
        return _ENV_SRC_CACHE[cache_key]
    ns = {}
    exec(compile(code, f"<gym_env:{cache_key}>", "exec"), ns)
    cls = ns[env_name]
    _ENV_SRC_CACHE[cache_key] = cls
    return cls


def make_env(ability, env_dir):
    """[兼容旧接口] 仅按 ability 从 env_dir/<mid>.py 文件实例化环境。

    ability: 'codegym_v1@<code_id>__<EnvName>@{json参数}'
    env.from_env_str 期望 '<EnvName>@{json参数}'
    """
    parts = ability.split("@", 2)
    mid, params_json = parts[1], parts[2]
    cls, env_name = load_env_class(env_dir, mid)
    inner = f"{env_name}@{params_json}"
    return cls.from_env_str(inner)


def make_env_from_item(item, env_dir=None):
    """根据一条数据样本实例化一个全新环境（兼容两种数据源）。

    优先使用样本自带的环境源码 extra_info['gym_env']（HF 完整数据集），
    回退到 env_dir/<mid>.py 文件（example 小数据集）。
    """
    ability = item["ability"]
    parts = ability.split("@", 2)
    mid, params_json = parts[1], parts[2]
    env_name = "__".join(mid.split("__")[1:])  # 去掉 code_id 前缀
    src = (item.get("extra_info") or {}).get("gym_env")
    if src:
        cls = load_env_class_from_source(src, env_name, mid)
    else:
        cls, env_name = load_env_class(env_dir, mid)
    inner = f"{env_name}@{params_json}"
    return cls.from_env_str(inner)


def _normalize_prompt(item):
    """把 parquet 读出的 prompt(numpy.ndarray[dict]) 规整为 list[dict]，供 apply_chat_template 使用。"""
    p = item.get("prompt")
    if p is not None and not isinstance(p, list):
        p = list(p)
    if p is not None:
        p = [dict(m) for m in p]
    item["prompt"] = p
    return item


def _iter_raw_records(data_path):
    """从 jsonl 文件 / parquet 文件 / 含 parquet(或 jsonl) 的目录 逐条产出原始样本 dict。"""
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        if not files:
            files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not files:
            raise FileNotFoundError(f"目录下未找到 .parquet/.jsonl 文件: {data_path}")
        for fp in files:
            yield from _iter_raw_records(fp)
    elif data_path.endswith(".parquet"):
        import pandas as pd  # 延迟导入，仅 parquet 路径需要
        df = pd.read_parquet(data_path)
        for rec in df.to_dict(orient="records"):
            yield rec
    elif data_path.endswith(".jsonl"):
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        raise ValueError(f"不支持的数据文件类型(需 .jsonl/.parquet 或目录): {data_path}")


def load_dataset(data_file, env_dir=None, max_solve_round=10**9,
                 easy_first=True, max_instances=None):
    """加载训练样本，自动识别两种数据源：

    1) example 小数据集：jsonl 文件 + env_dir/<mid>.py 环境文件
    2) HF 完整数据集：parquet 文件/目录，环境源码内置于 extra_info['gym_env']
       （如 data/codegym_hf/task_en_instruction_en_env，无需 env_dir）

    过滤：solve_fc_round <= max_solve_round 且环境可加载（源码自带或 .py 存在）。
    easy_first 按难度升序；max_instances 截断为前 N 条（配合 easy_first 取最简单 N 条）。
    """
    rows = []
    for item in _iter_raw_records(data_file):
        item = _normalize_prompt(item)
        ei = item.get("extra_info") or {}
        if ei.get("solve_fc_round", 10**9) > max_solve_round:
            continue
        # 环境可用性：自带源码 或 磁盘 .py 存在
        if not ei.get("gym_env"):
            mid = item["ability"].split("@", 2)[1]
            if env_dir is None or not os.path.exists(os.path.join(env_dir, mid + ".py")):
                continue
        rows.append(item)
        if max_instances and not easy_first and len(rows) >= max_instances:
            break
    if easy_first:
        rows.sort(key=lambda r: (r.get("extra_info") or {}).get("solve_fc_round", 10**9))
    if max_instances:
        rows = rows[:max_instances]
    return rows


# ----------------------------------------------------------------------
# 2. function-call 解析
# ----------------------------------------------------------------------
_FC_PATTERNS = [
    r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>",
    r"<tool_call>(.*?)</tool_call>",
    r"```(?:json)?\s*(.*?)```",
]


def parse_action(text):
    """从模型输出中解析出 {name, parameters}，解析失败返回 None。"""
    blob = None
    for pat in _FC_PATTERNS:
        m = re.search(pat, text, re.S)
        if m:
            blob = m.group(1)
            break
    if blob is None:
        m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        blob = m.group(1) if m else None
    if blob is None:
        return None
    try:
        obj = json.loads(blob.strip())
    except Exception:
        return None
    if isinstance(obj, list):
        obj = obj[0] if obj else None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return {"name": obj["name"], "parameters": obj.get("parameters", {})}


# ----------------------------------------------------------------------
# 3. 多轮 rollout
# ----------------------------------------------------------------------
# few-shot 格式引导：CodeGym 的原始 prompt 是为已微调模型设计的，
# 未微调的 Qwen2.5-3B-Instruct 不认 <|FunctionCallBegin|> 协议、倾向写散文，
# 因此在 system 末尾追加一段强格式说明 + 通用示例（不泄漏任何任务答案），
# 用 in-context 方式引导其立即产出合法 function call。
FEWSHOT_SUFFIX = """

========================  HOW TO RESPOND (READ CAREFULLY)  ========================
You are NOT allowed to write prose, explanations, reasoning, or code.
At EVERY step you must output EXACTLY ONE function call and nothing else,
using this exact format (a JSON list with one object, wrapped by the markers):

<|FunctionCallBegin|>[{"name": "<FunctionName>", "parameters": {<args>}}]<|FunctionCallEnd|>

Rules:
- Always START by calling Observe to read the current state:
  <|FunctionCallBegin|>[{"name": "Observe", "parameters": {}}]<|FunctionCallEnd|>
- After each call, STOP and wait for the "Observation:" message, then issue the next call.
- When you are confident, submit via the Done function with your final answer.
- Output ONLY the marker-wrapped JSON. No extra words before or after.
=================================================================================="""


def build_messages(base_messages, use_fewshot):
    msgs = [dict(m) for m in base_messages]
    if use_fewshot and msgs and msgs[0].get("role") == "system":
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = msgs[0]["content"] + FEWSHOT_SUFFIX
    return msgs


@torch.no_grad()
def rollout_one(model, tok, base_messages, env, cfg, device, debug=False, record=False):
    """跑一条多轮轨迹。返回 (turn_samples, stats)。

    turn_samples: list of (prompt_ids[1D cpu], response_ids[1D cpu])
                  每个 assistant turn 一个样本，整条轨迹共享同一 advantage。
    stats: dict，含 task_reward(纯 0/1)、shaped(训练用)、fmt、action、done 等；
           若 record=True，额外含 trace=[每轮的模型输出/动作/环境返回]。
    """
    messages = build_messages(base_messages, cfg.few_shot)
    turn_samples = []
    trace = []
    fail_cnt = 0
    n_format_ok = 0
    n_action_ok = 0
    called_done = False
    n_rounds = 0
    done = False

    for _ in range(cfg.max_rounds):
        prompt_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        attn = torch.ones_like(prompt_ids)
        out = model.generate(
            prompt_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        resp_ids = out[0, prompt_ids.shape[1]:]
        text = tok.decode(resp_ids, skip_special_tokens=True)
        n_rounds += 1
        turn_samples.append((prompt_ids[0].cpu(), resp_ids.cpu()))
        messages.append({"role": "assistant", "content": text})

        act = parse_action(text)
        if act is None:
            fail_cnt += 1
            obs = ("Error: cannot parse a function call. Wrap exactly one JSON "
                   '{"name":..., "parameters":{...}} inside '
                   "<|FunctionCallBegin|>[ ... ]<|FunctionCallEnd|>.")
            if fail_cnt >= cfg.max_parse_fail:
                messages.append({"role": "user", "content": obs[:cfg.max_obs_chars]})
                break
        else:
            n_format_ok += 1
            if act["name"] == "Done":
                called_done = True
            try:
                status, obs = env.step(json.dumps({"name": act["name"],
                                                   "parameters": act["parameters"]}))
                if not str(obs).startswith("Error"):
                    n_action_ok += 1
            except Exception as e:  # noqa: BLE001
                obs = f"Error during environment step: {e}"
        messages.append({"role": "user",
                         "content": f"Observation: {str(obs)[:cfg.max_obs_chars]}"})
        if record:
            trace.append({
                "round": n_rounds,
                "model_output": text.strip(),
                "parsed_action": (None if act is None else act["name"]),
                "observation": str(obs)[:cfg.max_obs_chars],
            })

        if bool(getattr(env, "finished", False)):
            done = True
            break

    task_reward = float(getattr(env, "reward", 0.0))
    fmt_rate = n_format_ok / max(1, n_rounds)
    action_rate = n_action_ok / max(1, n_rounds)

    # 主奖励 = CodeGym 原生 0/1；可选叠加小的工具使用塑形帮助冷启动
    shaped = task_reward
    if cfg.shaping:
        shaped += (cfg.w_format * fmt_rate
                   + cfg.w_action * action_rate
                   + (cfg.w_done if called_done else 0.0))

    if debug:
        print("\n----- [debug] 首条 rollout 对话 -----")
        for m in messages[2:]:  # 跳过 system+user 初始 prompt
            print(f"  [{m['role']}] {str(m['content'])[:200]}")
        print(f"  -> task_reward={task_reward} shaped={shaped:.3f} "
              f"fmt={fmt_rate:.2f} action={action_rate:.2f} done={done}")
        print("-------------------------------------\n")

    stats = {"task_reward": task_reward, "shaped": shaped, "n_rounds": n_rounds,
             "fmt": fmt_rate, "action": action_rate, "done": float(done)}
    if record:
        stats["trace"] = trace
    return turn_samples, stats


# ----------------------------------------------------------------------
# 4. logprob 计算 & GRPO 更新
# ----------------------------------------------------------------------
def per_token_logprob(model, prompt_ids, resp_ids, device, max_seq_len=2048):
    """计算 resp_ids 每个 token 在给定 prompt 下的 logprob，返回 1D 张量(带梯度)。

    显存优化：
    - 总长超 max_seq_len 时，从左侧裁剪 prompt（response 必须完整保留）；
    - 用 logsumexp 技巧避免把整个 [seq_len, vocab] 转 float32（vocab≈150k，极占显存）。
    """
    # 截断：response 完整，prompt 留近端
    if prompt_ids.shape[0] + resp_ids.shape[0] > max_seq_len:
        keep_prompt = max(1, max_seq_len - resp_ids.shape[0])
        prompt_ids = prompt_ids[-keep_prompt:]
    inp = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(device)
    logits = model(inp).logits[0, :-1]                  # 保持原 dtype(bf16)，不整体转 float
    targets = inp[0, 1:]
    # log p(target) = logit[target] - logsumexp(logits)，仅 logsumexp 走 float
    tgt_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    lse = torch.logsumexp(logits.float(), dim=-1)
    tok_logp = tgt_logit - lse
    return tok_logp[prompt_ids.shape[0] - 1:]           # 只取 response 部分


def grpo_update(model, optimizer, groups, cfg, device):
    """对一批 rollout 数据做 GRPO 更新。groups: list[ list[traj_dict] ]，每个内层 list 是一个 prompt 的 G 条轨迹。"""
    # 1) 组内归一化 advantage（每条轨迹一个标量，摊到该轨迹所有 turn）
    samples = []  # (prompt_ids, resp_ids, advantage)
    all_rewards = []
    for trajs in groups:
        rs = np.array([t["reward"] for t in trajs], dtype=np.float32)
        all_rewards.extend(rs.tolist())
        adv = (rs - rs.mean()) / (rs.std() + 1e-4)
        for t, a in zip(trajs, adv):
            for (p_ids, r_ids) in t["turns"]:
                if r_ids.numel() > 0:
                    samples.append((p_ids, r_ids, float(a)))

    if not samples:
        return {"loss": 0.0, "n_samples": 0, "mean_reward": float(np.mean(all_rewards or [0]))}

    # 2) 旧策略 logprob（固定，用于 PPO ratio）；存到 CPU 省显存
    old_logps = []
    with torch.no_grad():
        for p_ids, r_ids, _ in samples:
            old_logps.append(per_token_logprob(model, p_ids, r_ids, device).detach().cpu())

    # 3) PPO-clip 多个 epoch
    total_loss = 0.0
    for _ in range(cfg.ppo_epochs):
        optimizer.zero_grad()
        loss_accum = 0.0
        for si, ((p_ids, r_ids, adv), old_lp) in enumerate(zip(samples, old_logps)):
            new_lp = per_token_logprob(model, p_ids, r_ids, device)
            old_lp = old_lp.to(device)
            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            pg = -torch.min(unclipped, clipped).mean()

            if cfg.kl_beta > 0:
                with model.disable_adapter(), torch.no_grad():
                    ref_lp = per_token_logprob(model, p_ids, r_ids, device)
                kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1  # k3 估计
                pg = pg + cfg.kl_beta * kl.mean()

            (pg / len(samples)).backward()
            loss_accum += float(pg.detach()) / len(samples)
            del new_lp, old_lp, ratio, pg
            if (si + 1) % 16 == 0:
                torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        optimizer.step()
        total_loss = loss_accum
    torch.cuda.empty_cache()

    return {"loss": total_loss, "n_samples": len(samples),
            "mean_reward": float(np.mean(all_rewards))}


# ----------------------------------------------------------------------
# 4.5 轨迹记录（每隔若干 step 落盘一条完整 rollout，供人工查看）
# ----------------------------------------------------------------------
def write_rollout_records(md_path, jsonl_path, step, records):
    """把若干条完整 rollout 以 markdown(易读) + jsonl(机读) 双格式追加落盘。

    records: list of dict，每个含 ability / stats(含 trace)。
    """
    # jsonl：机读
    with open(jsonl_path, "a") as jf:
        for rec in records:
            jf.write(json.dumps({"step": step, **rec}, ensure_ascii=False) + "\n")

    # markdown：人读
    with open(md_path, "a") as mf:
        mf.write(f"\n\n{'='*80}\n# Step {step} 训练轨迹样本\n{'='*80}\n")
        for ri, rec in enumerate(records, 1):
            st = rec["stats"]
            env_name = rec["ability"].split("@")[1].split("__")[-1]
            params = rec["ability"].split("@", 2)[2]
            verdict = "✅ 解对 (reward=1)" if st["task_reward"] > 0 else "❌ 未解对 (reward=0)"
            mf.write(f"\n## 样本 {ri}：`{env_name}`  {verdict}\n\n")
            mf.write(f"- **任务初始化**：`{params}`\n")
            mf.write(f"- **得分**：task_reward(纯0/1)=**{st['task_reward']:.0f}**  | "
                     f"shaped(训练用)={st['shaped']:.3f}\n")
            mf.write(f"- **统计**：共 {st['n_rounds']} 轮 | 格式正确率 fmt={st['fmt']:.2f} | "
                     f"成功调用率={st['action']:.2f} | 是否提交Done={'是' if st['done'] else '否'}\n\n")
            mf.write("| 轮次 | 模型输出（function call） | 环境返回（observation） |\n")
            mf.write("|---|---|---|\n")
            for t in st.get("trace", []):
                mo = t["model_output"].replace("\n", " ").replace("|", "\\|")[:160]
                ob = str(t["observation"]).replace("\n", " ").replace("|", "\\|")[:120]
                mf.write(f"| {t['round']} | `{mo}` | `{ob}` |\n")
    print(f"[record] step {step}: 已记录 {len(records)} 条完整 rollout → {md_path}")


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
DEFAULT_MODEL_PATH = (
    "/home2/zydc/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data_file", default=os.path.join(PROJECT_ROOT, "example/training_instance.jsonl"),
                    help="训练样本来源：jsonl 文件 / parquet 文件 / 含 parquet 的目录"
                         "（如 data/codegym_hf/task_en_instruction_en_env，环境源码内置无需 env_dir）")
    ap.add_argument("--env_dir", default=os.path.join(PROJECT_ROOT, "example/example_envs"),
                    help="example 小数据集的环境 .py 目录；用 HF parquet 完整数据集时可忽略")
    ap.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "training/output"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_rounds", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--max_obs_chars", type=int, default=800)
    ap.add_argument("--max_parse_fail", type=int, default=3)
    ap.add_argument("--max_solve_round", type=int, default=15,
                    help="只用 solve_fc_round <= 该值的简单任务")
    ap.add_argument("--max_instances", type=int, default=0,
                    help="加载的训练样本上限(0=不限)；配合 easy_first 取最简单的前 N 条，"
                         "用于在 8 万条完整数据集上快速起步")
    ap.add_argument("--shaping", type=int, default=1,
                    help="1=叠加工具使用塑形奖励助冷启动; 0=纯 CodeGym 0/1 奖励")
    ap.add_argument("--few_shot", type=int, default=1,
                    help="1=在 system 末尾注入格式引导(未微调模型必需)")
    ap.add_argument("--w_format", type=float, default=0.15)
    ap.add_argument("--w_action", type=float, default=0.15)
    ap.add_argument("--w_done", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ppo_epochs", type=int, default=1)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_beta", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="打印首条 rollout 对话")
    ap.add_argument("--record_every", type=int, default=5,
                    help="每隔多少 step 完整记录一批 rollout（模型输出/动作/环境返回/得分）")
    ap.add_argument("--record_n", type=int, default=2,
                    help="每个被记录的 step 记录多少条完整 rollout")
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()

    if cfg.smoke:
        cfg.steps, cfg.prompts_per_step, cfg.group_size = 2, 2, 4
        cfg.max_rounds, cfg.max_new_tokens = 8, 96
        cfg.max_solve_round = 12

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = "cuda"

    print(f"[init] loading tokenizer & model: {cfg.model_path}")
    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True)
    model.config.use_cache = True

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_r * 2, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    data = load_dataset(cfg.data_file, cfg.env_dir,
                        max_solve_round=cfg.max_solve_round, easy_first=True,
                        max_instances=(cfg.max_instances or None))
    print(f"[init] usable training instances: {len(data)} "
          f"(solve_fc_round <= {cfg.max_solve_round})")
    print(f"[init] config: steps={cfg.steps} prompts/step={cfg.prompts_per_step} "
          f"group={cfg.group_size} max_rounds={cfg.max_rounds} shaping={cfg.shaping}")

    metrics_path = os.path.join(cfg.output_dir, "metrics.jsonl")
    mf = open(metrics_path, "w")
    rec_md_path = os.path.join(cfg.output_dir, "rollout_records.md")
    rec_jsonl_path = os.path.join(cfg.output_dir, "rollout_records.jsonl")
    open(rec_md_path, "w").close()      # 清空旧记录
    open(rec_jsonl_path, "w").close()

    for step in range(1, cfg.steps + 1):
        batch = random.sample(data, min(cfg.prompts_per_step, len(data)))
        groups = []
        t0 = time.time()
        step_task, step_shaped, step_rounds, step_fmt, step_done = [], [], [], [], []
        # 每隔 record_every 步（含第 1 步）完整记录若干条 rollout
        do_record = (step == 1) or (step % cfg.record_every == 0)
        step_records = []

        model.eval()
        for bi, item in enumerate(batch):
            trajs = []
            for gi in range(cfg.group_size):
                env = make_env_from_item(item, cfg.env_dir)
                dbg = cfg.debug and step == 1 and bi == 0 and gi == 0
                rec = do_record and bi == 0 and gi < cfg.record_n
                turns, st = rollout_one(
                    model, tok, item["prompt"], env, cfg, device, debug=dbg, record=rec)
                trajs.append({"turns": turns, "reward": st["shaped"]})
                step_task.append(st["task_reward"])
                step_shaped.append(st["shaped"])
                step_rounds.append(st["n_rounds"])
                step_fmt.append(st["fmt"])
                step_done.append(st["done"])
                if rec:
                    step_records.append({"ability": item["ability"], "stats": st})
            groups.append(trajs)

        if do_record and step_records:
            write_rollout_records(rec_md_path, rec_jsonl_path, step, step_records)

        model.train()
        info = grpo_update(model, optimizer, groups, cfg, device)

        log = {
            "step": step,
            "task_reward": round(float(np.mean(step_task)), 4),
            "solve_rate": round(float(np.mean([r > 0 for r in step_task])), 4),
            "shaped_reward": round(float(np.mean(step_shaped)), 4),
            "mean_rounds": round(float(np.mean(step_rounds)), 2),
            "format_ok": round(float(np.mean(step_fmt)), 3),
            "done_rate": round(float(np.mean(step_done)), 3),
            "loss": round(info["loss"], 5),
            "n_samples": info["n_samples"],
            "sec": round(time.time() - t0, 1),
        }
        print(f"[step {step:>3}/{cfg.steps}] task_R={log['task_reward']:.3f} "
              f"solve={log['solve_rate']:.3f} shaped={log['shaped_reward']:.3f} "
              f"rounds={log['mean_rounds']:.1f} fmt={log['format_ok']:.2f} "
              f"done={log['done_rate']:.2f} loss={log['loss']:.4f} "
              f"n={log['n_samples']} ({log['sec']}s)")
        mf.write(json.dumps(log) + "\n")
        mf.flush()

    mf.close()
    save_dir = os.path.join(cfg.output_dir, "lora_adapter")
    model.save_pretrained(save_dir)
    print(f"[done] LoRA adapter saved to {save_dir}")
    print(f"[done] metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
