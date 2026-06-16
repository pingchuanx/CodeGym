# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

"""
veRL ↔ CodeGym 在线 rollout 桥接 shim 服务
=========================================

把 CodeGym 的「两级 manager + session」在线环境服务，适配成 veRL
(`verl/earl_http_env`) 的 ``GenericHttpEnvTool`` 所期望的「单级 env_id + RESTful」
协议，从而让 veRL 的 multi-turn agentic GRPO/PPO 训练可以零改动地驱动 CodeGym 环境。

为什么需要这一层（两边协议差异）：
  * CodeGym：先向 manager(:8000) ``/get_instance`` 要一个 worker 端口，再按
    ``session_id`` 调 worker 的 ``/start /step``；``reward``/``is_done`` 是**独立**
    的 GET 端点，不在 step 返回里。
  * veRL  ：单个 ``base_url``，生命周期 ``create -> step* -> close``，且期望从
    **同一个** step 响应里就能抽出 ``observation/reward/done``。

本 shim 的做法（保持极薄）：
  * 直接复用 ``example_rollout_env.codegym_v1``（即 ``OnlineFcGymEnv``）——它已经封装好
    了「找 manager 要端口 + 建 session + 路由 + 重试」的全部复杂度，对外只暴露
    ``from_env_str / step / finished / reward / release``。
  * shim 自己维护 ``env_id -> codegym_v1 实例`` 的映射，把 veRL 的 env_id 协议映射过去。
  * 在 ``/step`` 里调用 ``env.step`` 后，**顺手回填** ``env.reward`` 与 ``env.finished``，
    拼进返回体，正好命中 veRL adapter 默认的 ``extract_reward/extract_done``。

环境与启动：
  * 用 CodeGym 的 ``online_server`` conda 环境运行（含 fastapi/uvicorn/requests）。
  * 前提：CodeGym 的 env manager 已启动（``python online_server/env_instance_manager.py
    --workers N``），且 ``SERVER_IP_ADDRESS`` 指向 manager 主机（默认 ``::`` 本机 IPv6）。
  * 启动 shim：
        cd online_server
        SERVER_IP_ADDRESS=:: BRIDGE_PORT=8200 \
            python verl_bridge_server.py
    或用 uvicorn：
        uvicorn verl_bridge_server:app --host :: --port 8200

  * 然后在 veRL 侧用 ``online_server/configs/codegym.yaml`` 把 ``base_url`` 指到
    ``http://<shim_host>:8200`` 即可。

注意：
  * CodeGym 客户端 ``get_one_instance`` 在 manager 不可达时会无限重试，因此 ``/create``
    加了 ``CREATE_TIMEOUT`` 兜底，超时返回 504（底层重试线程为 daemon，进程退出即回收）。
  * 每条训练样本的环境由其 ``ability`` 字符串
    （``codegym_v1@<code_id>__<EnvName>@<json任务参数>``）决定，需由 veRL 的 agent loop
    在调用工具 ``create`` 时，通过 ``create_kwargs.create_payload`` 把该串透传进来
    （字段名 ``env_str`` 或 ``ability`` 均可）。
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request

# 复用已验证的 CodeGym 在线环境客户端（内部处理 manager+session+重试）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_rollout_env import codegym_v1  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - [verl_bridge] %(message)s")
logger = logging.getLogger(__name__)

# /create 调用 CodeGym 客户端建环境的超时（秒）。manager 不可达时客户端会无限重试，
# 这里兜底避免单个 HTTP 请求永久挂起。
CREATE_TIMEOUT = float(os.environ.get("BRIDGE_CREATE_TIMEOUT", "120"))
# 单步 env.step 的超时（秒），透传给 CodeGym 客户端。
STEP_TIMEOUT = float(os.environ.get("BRIDGE_STEP_TIMEOUT", "30"))

app = FastAPI(title="CodeGym↔veRL bridge")

# env_id -> {"env": codegym_v1 实例, "initial_obs": str}
_ENVS: Dict[str, Dict[str, Any]] = {}
_LOCK = asyncio.Lock()


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
_ENV_STR_KEYS = ("env_str", "ability", "env", "task", "env_id_str")
_ID_KEYS = ("id", "env_id", "instance_id", "envId")


def _pick(payload: Dict[str, Any], keys) -> Optional[str]:
    """从 payload 中按优先级取第一个非空字符串字段。"""
    if not isinstance(payload, dict):
        return None
    for k in keys:
        v = payload.get(k)
        if v:
            return v
    return None


def _resolve_env_str(payload: Dict[str, Any]) -> Optional[str]:
    """从 veRL 的 create payload 中解析出 CodeGym 的 env_str（= dataset 的 ability）。

    兼容把任务串嵌套在 ``create_payload`` / ``reset_payload`` 里的情形。
    """
    direct = _pick(payload, _ENV_STR_KEYS)
    if direct:
        return direct
    for nested_key in ("create_payload", "reset_payload", "create_kwargs"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            got = _resolve_env_str(nested)
            if got:
                return got
    return None


def _get_entry(env_id: Optional[str]) -> Dict[str, Any]:
    if not env_id or env_id not in _ENVS:
        raise HTTPException(status_code=404, detail=f"unknown env id: {env_id!r}")
    return _ENVS[env_id]


# ----------------------------------------------------------------------
# 端点：veRL BaseHttpEnvClient 期望的协议
# ----------------------------------------------------------------------
@app.post("/create")
async def create(request: Request):
    """创建一个 CodeGym 环境 session，返回 env_id 与初始 observation。"""
    payload = await _safe_json(request)
    env_str = _resolve_env_str(payload)
    if not env_str:
        raise HTTPException(
            status_code=400,
            detail=("missing env_str/ability in create payload; expected a CodeGym "
                    "ability like 'codegym_v1@<code_id>__<EnvName>@{json}'"))

    def _make():
        env = codegym_v1.from_env_str(env_str)
        return env, (getattr(env, "initial_obs", "") or "")

    try:
        env, initial_obs = await asyncio.wait_for(asyncio.to_thread(_make), CREATE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("create timed out (%.0fs) for env_str=%s", CREATE_TIMEOUT, env_str)
        raise HTTPException(status_code=504, detail="CodeGym env create timed out")
    except Exception as e:  # noqa: BLE001
        logger.exception("create failed for env_str=%s", env_str)
        raise HTTPException(status_code=500, detail=f"create failed: {e}")

    env_id = str(uuid.uuid4())
    async with _LOCK:
        _ENVS[env_id] = {"env": env, "initial_obs": str(initial_obs)}
    logger.info("created env_id=%s for %s", env_id, env_str)
    return {"env_id": env_id, "id": env_id, "observation": str(initial_obs),
            "reward": 0.0, "done": False}


@app.post("/reset")
async def reset(request: Request):
    """重置：CodeGym 在 /start 时已完成 reset，这里返回缓存的初始 observation。"""
    payload = await _safe_json(request)
    entry = _get_entry(_pick(payload, _ID_KEYS))
    return {"observation": entry["initial_obs"], "reward": 0.0, "done": False}


@app.post("/step")
async def step(request: Request):
    """执行一步动作，并回填 CodeGym 的 reward/done 到同一响应中。"""
    payload = await _safe_json(request)
    entry = _get_entry(_pick(payload, _ID_KEYS))
    env = entry["env"]

    action = payload.get("action")
    if action is None:
        raise HTTPException(status_code=400, detail="missing 'action' in step payload")
    # CodeGym 期望动作是 JSON 字符串 {"name":..,"parameters":..}；非字符串则序列化。
    if not isinstance(action, str):
        action = json.dumps(action, ensure_ascii=False)

    def _do_step():
        status, obs = env.step(action, STEP_TIMEOUT)
        return status, obs, bool(getattr(env, "finished", False)), float(getattr(env, "reward", 0.0))

    try:
        status, obs, done, reward = await asyncio.to_thread(_do_step)
    except Exception as e:  # noqa: BLE001
        logger.exception("step failed")
        raise HTTPException(status_code=500, detail=f"step failed: {e}")

    return {"observation": str(obs), "reward": reward, "done": done, "status": status}


@app.post("/close")
async def close(request: Request):
    """释放环境 session（归还 manager 实例 + 通知 worker stop）。"""
    payload = await _safe_json(request)
    env_id = _pick(payload, _ID_KEYS)
    async with _LOCK:
        entry = _ENVS.pop(env_id, None)
    if entry is not None:
        try:
            await asyncio.to_thread(entry["env"].release)
        except Exception as e:  # noqa: BLE001
            logger.warning("release failed for env_id=%s: %s", env_id, e)
    return {"closed": True, "id": env_id}


@app.get("/observation")
async def observation(id: Optional[str] = None, env_id: Optional[str] = None):
    entry = _get_entry(id or env_id)
    return {"observation": entry["initial_obs"]}


@app.get("/available_actions")
async def available_actions(id: Optional[str] = None, env_id: Optional[str] = None):
    # CodeGym 的可用动作由 prompt 内嵌的函数文档描述，服务端不枚举 → 返回空列表。
    _get_entry(id or env_id)
    return {"available_actions": []}


@app.get("/health")
async def health():
    return {"status": "ok", "active_envs": len(_ENVS)}


async def _safe_json(request: Request) -> Dict[str, Any]:
    """容错解析请求体为 dict；空体或非 JSON 返回 {}。"""
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("BRIDGE_HOST", "::")
    port = int(os.environ.get("BRIDGE_PORT", "8200"))
    logger.info("starting CodeGym↔veRL bridge on [%s]:%d "
                "(CodeGym manager via SERVER_IP_ADDRESS=%s)",
                host, port, os.environ.get("SERVER_IP_ADDRESS", "::"))
    uvicorn.run(app, host=host, port=port)
