#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeGym 端到端演示脚本 (demo_run.py)
====================================

本脚本演示 CodeGym 项目的核心运行逻辑，分为两部分：

  Part A —— 离线验证 Gym 环境逻辑
      动态加载若干 example 环境，运行其参考 solve()（一条由多轮原子工具
      调用组成的轨迹），校验最终 reward==1 并统计调用步数。这对应论文中
      "每道静态代码题被改写为一组带文档的工具 + 一个可自动判奖的环境"。

  Part B —— 端到端在线服务 rollout（录制-重放，无需 LLM）
      1) 本地实例化环境并运行参考 solve()，录制 (action, observation) 轨迹；
      2) 启动 / 连接 env_instance_manager，申请一个 env_server 实例；
      3) POST /start 初始化会话；
      4) 把录制的参考动作序列逐步 POST /step 重放到服务端；
      5) GET /is_done、GET /reward 读取终止信号与奖励；
      6) POST /stop + GET /release_instance 释放资源。
      （真实 RL 训练中，动作序列由策略模型在线生成；此处用参考动作替代，
        以在没有 LLM 的情况下完整验证 rollout 链路与奖励计算。）

用法:
    python demo_run.py                # 跑 Part A + Part B（自动拉起 manager）
    python demo_run.py --offline      # 只跑 Part A
    python demo_run.py --no-manager   # 跑 Part B 但不自动拉起 manager
                                      #   （需自行先启动 env_instance_manager）
"""
import os
import sys
import json
import time
import uuid
import importlib.util
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
ENV_SOURCE_DIR = os.path.join(THIS_DIR, "example_envs")
ONLINE_SERVER_DIR = os.path.join(PROJECT_ROOT, "online_server", "online_server")

try:
    import requests
except ImportError:
    requests = None


# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------
def load_env_class(env_file):
    """从环境 .py 文件动态加载环境类，返回 (类对象, 类名)。

    文件名形如 `Code_Contests_26156_I__HouseRobberEnv.py`，类名为去掉
    `<code_id>__` 前缀与 `.py` 后缀后的 `HouseRobberEnv`。
    """
    path = os.path.join(ENV_SOURCE_DIR, env_file)
    spec = importlib.util.spec_from_file_location(f"envmod_{env_file[:-3]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls_name = "__".join(os.path.basename(env_file).split("__")[1:])[:-3]
    return getattr(mod, cls_name), cls_name


def record_reference_trajectory(env_cls, cls_name, params):
    """本地实例化环境并运行参考 solve()，录制 (action, observation) 序列。

    通过临时替换实例的 step 方法来捕获 solve() 内部发出的每一个动作，
    既不修改环境源码，也不触碰任何内部状态。
    """
    env = env_cls.from_env_str(f"{cls_name}@{json.dumps(params)}")
    trajectory = []
    original_step = env.step

    def patched_step(action):
        status, obs = original_step(action)
        trajectory.append((action, obs))
        return status, obs

    env.step = patched_step
    final_msg = env.solve()
    return trajectory, final_msg, env.reward, env.finished


# ----------------------------------------------------------------------
# Part A —— 离线验证 Gym 环境逻辑
# ----------------------------------------------------------------------
def part_a_offline():
    print("=" * 72)
    print("Part A — 离线验证 Gym 环境逻辑（参考 solve 多轮工具调用 → 自动判奖）")
    print("=" * 72)

    demo_files = [
        "Code_Contests_26156_I__HouseRobberEnv.py",
        "Code_Contests_33538_I__StockProfitEnv.py",
        "Codeforces_38402_I__MaxGCDAfterDivideEnv.py",
        "Leetcode_22783_I__HouseRobberEnv.py",
        "Code_Contests_10107_I__LargestNumberEnv.py",
    ]

    passed = 0
    for f in demo_files:
        path = os.path.join(ENV_SOURCE_DIR, f)
        if not os.path.exists(path):
            print(f"\n[SKIP] 找不到 {f}")
            continue
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=60,
            )
            out = proc.stdout.strip()
            ok = "reward=1" in out
            passed += int(ok)
            print(f"\n[{'PASS' if ok else 'FAIL'}] {f}")
            for line in out.splitlines():
                print("    ", line)
            if not ok and proc.stderr.strip():
                print("     [stderr]", proc.stderr.strip()[:300])
        except Exception as e:  # noqa: BLE001
            print(f"\n[ERROR] {f}: {e}")

    print(f"\nPart A 小结：{passed}/{len(demo_files)} 个环境的参考解通过（reward=1）\n")
    return passed == len(demo_files)


# ----------------------------------------------------------------------
# Part B —— 端到端在线服务 rollout
# ----------------------------------------------------------------------
MANAGER_HOSTS = ["http://127.0.0.1:8000", "http://[::1]:8000"]


def start_manager():
    """以正确的工作目录拉起 env_instance_manager（少量 worker），返回子进程句柄。"""
    env = dict(os.environ)
    env["SERVER_PUBLIC_HOST"] = "127.0.0.1"  # 避免实例对外探测公网 IP
    log = open(os.path.join(THIS_DIR, "manager.log"), "w")
    proc = subprocess.Popen(
        [sys.executable, "env_instance_manager.py", "--workers", "4"],
        cwd=ONLINE_SERVER_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    return proc


def manager_get_instance(timeout_s=150):
    """轮询 manager 直至成功拿到一个实例，返回 (manager_host, {uid, port})。"""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        for host in MANAGER_HOSTS:
            try:
                r = requests.get(f"{host}/get_instance", timeout=5)
                if r.status_code == 200:
                    return host, r.json()
                last_err = f"{host} -> HTTP {r.status_code}: {r.text[:120]}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{host} -> {e}"
        time.sleep(2)
    raise RuntimeError(f"无法从 env_instance_manager 获取实例: {last_err}")


def part_b_online(auto_manager=True):
    print("=" * 72)
    print("Part B — 端到端在线服务 rollout（录制参考动作 → HTTP 重放 → 读奖励）")
    print("=" * 72)

    if requests is None:
        print("[ERROR] 未安装 requests，无法运行 Part B")
        return False

    env_file = "Code_Contests_26156_I__HouseRobberEnv.py"
    code_id_envname = env_file[:-3]  # Code_Contests_26156_I__HouseRobberEnv
    params = {"n": 8, "gold": [2, 7, 9, 3, 1, 5, 8, 4]}

    # 1) 本地录制参考动作序列
    env_cls, cls_name = load_env_class(env_file)
    trajectory, final_msg, local_reward, local_done = record_reference_trajectory(
        env_cls, cls_name, params
    )
    print(f"\n[录制] 环境 {cls_name} 参考解共 {len(trajectory)} 步；"
          f"本地 done={local_done}, reward={local_reward}")
    print(f"[录制] 终止信息: {final_msg}")

    manager_proc = None
    try:
        if auto_manager:
            print("\n[manager] 正在拉起 env_instance_manager（--workers 4）…")
            manager_proc = start_manager()

        # 2) 申请一个 env_server 实例
        manager_host, inst = manager_get_instance()
        uid, port = inst["uid"], inst["port"]
        base = manager_host.rsplit(":", 1)[0]
        env_host = f"{base}:{port}"
        print(f"[实例] manager={manager_host} 分配 env_server={env_host} (uid={uid[:8]}…)")

        session_id = str(uuid.uuid4())
        env_str = f"{code_id_envname}@{json.dumps(params)}"

        # 3) POST /start（带重试，等待 env_server 的 uvicorn 子进程就绪）
        obs = None
        for _ in range(20):
            try:
                r = requests.post(
                    f"{env_host}/start",
                    json={"session_id": session_id, "env_str": env_str, "env_name": "codegym_v1"},
                    timeout=30,
                )
                if r.status_code == 200:
                    obs = r.json().get("observation")
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        if obs is None:
            print("[ERROR] /start 失败")
            return False
        print(f"[start] 会话已建立, observation={obs!r}")

        # 4) 逐步重放录制的参考动作
        mismatch = 0
        print("[step ] 开始重放参考动作序列…")
        for i, (action, local_obs) in enumerate(trajectory, 1):
            r = requests.post(
                f"{env_host}/step",
                json={"session_id": session_id, "action": action, "timeout": 30},
                timeout=35,
            ).json()
            server_obs = r.get("observation")
            same = (server_obs == local_obs)
            mismatch += int(not same)
            try:
                name = json.loads(action)["name"]
            except Exception:  # noqa: BLE001
                name = "?"
            if i <= 3 or i == len(trajectory):
                preview = (server_obs or "")[:48].replace("\n", " ")
                print(f"   step {i:>2}: {name:<16} obs一致={same}  server_obs={preview!r}")
            elif i == 4:
                print("   …")

        # 5) 读取终止信号与奖励
        done = requests.get(
            f"{env_host}/is_done", params={"session_id": session_id}, timeout=10
        ).json()["done"]
        reward = requests.get(
            f"{env_host}/reward", params={"session_id": session_id}, timeout=10
        ).json()["reward"]
        print(f"\n[结果] 服务端 is_done={done}, reward={reward}; "
              f"本地/服务端观测不一致步数={mismatch}")

        # 6) 释放资源
        requests.post(f"{env_host}/stop", json={"session_id": session_id}, timeout=10)
        requests.get(f"{manager_host}/release_instance", params={"uid": uid}, timeout=10)
        print("[清理] 已 stop 会话并 release 实例")

        ok = bool(done) and float(reward) == 1.0 and mismatch == 0
        print(f"\nPart B 小结：{'通过' if ok else '失败'}"
              f"（服务端复现了参考解并给出 reward=1）\n")
        return ok
    finally:
        if manager_proc is not None:
            print("[manager] 正在关闭 env_instance_manager…")
            manager_proc.terminate()
            try:
                manager_proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                manager_proc.kill()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    run_offline_only = "--offline" in sys.argv
    auto_manager = "--no-manager" not in sys.argv

    a_ok = part_a_offline()
    b_ok = True
    if not run_offline_only:
        b_ok = part_b_online(auto_manager=auto_manager)

    print("=" * 72)
    print(f"总结: Part A = {'通过' if a_ok else '失败'}; "
          f"Part B = {'通过' if (run_offline_only) else ('通过' if b_ok else '失败')}"
          f"{'（已跳过）' if run_offline_only else ''}")
    print("=" * 72)
    sys.exit(0 if (a_ok and b_ok) else 1)


if __name__ == "__main__":
    main()
