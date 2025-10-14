from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime, timedelta
import uuid
import asyncio
import logging
import dill as pickle
from server_config import get_class
import queue
import multiprocessing
import gc
import psutil
import os

# ---------------------- Configuration ----------------------
SESSION_TIMEOUT_MINUTES = 180 # 3 hours to expire
SAFETY_MARGIN = 2

lock = asyncio.Lock()

# set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# store environment and last used time
class Session:

    def __init__(self, env):
        self.env = env
        self.last_used = datetime.utcnow()


env_sessions: Dict[str, Session] = {}

# ---------------------- Model Definition ----------------------


class StepRequest(BaseModel):
    session_id: str
    action: str
    timeout: Optional[float] = 30


class StartRequest(BaseModel):
    session_id: Optional[str] = None
    env_str: Optional[str] = None
    env_name: str = None


class StopRequest(BaseModel):
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


# ---------------------- Periodic Cleanup Task ----------------------


async def cleanup_sessions():
    while True:
        await asyncio.sleep(60)
        async with lock:
            now = datetime.utcnow()
            to_delete = [
                sid for sid, sess in env_sessions.items()
                if now - sess.last_used > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
            ]
            for sid in to_delete:
                logger.info(f"Session {sid} expired. Cleaning up.")
                if hasattr(env_sessions[sid].env, "close"):
                    env_sessions[sid].env.close()
                del env_sessions[sid].env
                del env_sessions[sid]

# --------------------- Periodic GC ---------------------
async def periodic_gc(interval_sec=300):
    process = psutil.Process(os.getpid())
    while True:
        await asyncio.sleep(interval_sec)
        mem = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[GC] Triggering gc.collect()... Current RSS memory: {mem:.2f} MB")
        collected = gc.collect()
        mem_after = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[GC] Collected {collected} objects. Memory after GC: {mem_after:.2f} MB")


# define lifespan event handler
async def lifespan(app: FastAPI):
    # when starting
    cleanup_task = asyncio.create_task(cleanup_sessions())
    gc_task = asyncio.create_task(periodic_gc(300))
    logger.info("Started env service with session cleaner.")
    yield
    # when closing
    cleanup_task.cancel()
    gc_task.cancel()
    await cleanup_task
    await gc_task


# create FastAPI app with lifespan parameter
app = FastAPI(lifespan=lifespan)

# ---------------------- Interface Implementation ----------------------


@app.post("/start")
async def start(req: StartRequest):
    session_id = req.session_id or str(uuid.uuid4())
    env_name = req.env_name
    env_str = req.env_str
    prefix = env_str.split("@")[0] + '@'
    try:
        # logger.info(f"[Start] Started session {session_id}, env_name: {env_name}, env_str: {env_str}, prefix: {prefix}")
        env_cls = get_class(env_name, prefix)
        if env_cls is None:
            raise ValueError("No valid environment found.")
        if env_name.startswith("codegym"):
            env_str = "__".join(env_str.split("__")[1:])
        env = env_cls.from_env_str(env_str)
        obs = "successfully start"
        for name in ["_get_obs", "get_obs"]:
            if hasattr(env, name):
                obs = getattr(env, name)()
                break
        async with lock:
            env_sessions[session_id] = Session(env)
        logger.info(f"[Start] Started session {session_id}, env_name: {env_name}, env_str: {env_str}, prefix: {prefix} successed")
        return {"session_id": session_id, "observation": obs}
    except Exception as e:
        logger.exception(f"[Start] Failed to start session: {e}, env_str: {env_str}")
        raise HTTPException(status_code=500, detail=str(e))


def step_in_process(env_bytes, action, queue):
    try:
        import logging
        logger = logging.getLogger(__name__)
        # logger.info("[STEP] begin step in process")
        
        env = pickle.loads(env_bytes)
        # logger.info("[STEP] env load successed")
        
        status, observation = env.step(action)
        # logger.info(f"[STEP] env step successed, status={status}, observation preview={str(observation)[:100]}")
        
        updated_env_bytes = pickle.dumps(env)
        # logger.info("[STEP] env dump successed")

        queue.put((status, observation, updated_env_bytes))
        # logger.info("[STEP] queue put successed")
    except Exception as e:
        logger.exception("[STEP] Exception in step_in_process")
        queue.put(("error", str(e)))


class TimeoutException(Exception):
    pass


@app.post("/step")
async def step(req: StepRequest):
    total_time = max(3, req.timeout - SAFETY_MARGIN) # TODO: hard-coded here. May need different timeout mechanisms for different envs.
    async with lock:
        session = env_sessions.get(req.session_id)
        if not session:
            logger.info(f"[STEP] Session id {req.session_id} not found, maybe expired")
            raise HTTPException(status_code=404, detail=f"[STEP] Session id {req.session_id} not found, maybe expired")

        try:
            logger.info(f"[STEP] Step action: {req.action}, session_id: {req.session_id}")

            env_bytes = pickle.dumps(session.env)  # deepcopy env
            msg_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=step_in_process, args=(env_bytes, req.action, msg_queue))
            p.start()
            
            try:
                result = msg_queue.get(timeout=total_time)
            except queue.Empty:
                logger.info(
                    f"[STEP] Step action: {req.action}, session_id: {req.session_id}, fake action timeout, try to kill the process"
                )
                p.kill()
                logger.info(f"[STEP] Step action: {req.action}, session_id: {req.session_id}, kill process successed")
                result = ("error", "Step timed out.")
            finally:
                # ensure process resource cleanup
                p.join(timeout=SAFETY_MARGIN / 2)
                if p.is_alive():
                    logger.warning(f"[STEP] Process still alive, force kill {p.pid}")
                    p.kill()
                    p.join(timeout=1)
                try:
                    p.close()
                except Exception as e:
                    logger.warning(f"[STEP] Failed to close process: {e}")

                # ensure queue resource cleanup
                try:
                    msg_queue.close()
                    msg_queue.join_thread()
                except Exception as e:
                    logger.warning(f"[STEP] Failed to close queue: {e}")

                # manually trigger garbage collection
                gc.collect()

            if result[0] == "error":
                raise Exception(f"Step failed: {result[1]}")

            status, observation, updated_env_bytes = result
            session.env = pickle.loads(updated_env_bytes)
            session.last_used = datetime.utcnow()
            logger.info(f"[STEP] Step action: {req.action}, session_id: {req.session_id}, true action succeeded | observation = {observation}")
            return {"status": status, "observation": observation}

        except Exception as e:
            logger.exception(f"Step failed for session {req.session_id}, action: {req.action}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/is_done")
async def is_done(session_id: str):
    async with lock:
        logger.info(f"[IS_DONE] try session_id: {session_id}")
        session = env_sessions.get(session_id)
        if not session:
            logger.info(f"[IS_DONE] session_id: {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        session.last_used = datetime.utcnow()
        logger.info(f"[IS_DONE] successed session_id: {session_id}, done: {session.env.finished}")
        return {"done": session.env.finished}


@app.get("/is_success")
async def is_success(session_id: str):
    async with lock:
        session = env_sessions.get(session_id)
        if not session:
            logger.info(f"[IS_SUCCESS] session_id: {session_id} not found")
            raise HTTPException(status_code=404, detail="[IS_SUCCESS] Session not found")
        session.last_used = datetime.utcnow()
        return {"success": session.env.success()}


@app.get("/reward")
async def reward(session_id: str):
    async with lock:
        logger.info(f"[REWARD] try session_id: {session_id}")
        session = env_sessions.get(session_id)
        if not session:
            logger.info(f"[REWARD] session_id: {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        session.last_used = datetime.utcnow()
        logger.info(f"[REWARD] successed session_id: {session_id}, reward: {session.env.reward}")
        return {"reward": session.env.reward}


@app.post("/stop")
async def stop(req: StopRequest):
    async with lock:
        session = env_sessions.pop(req.session_id, None)
        if session:
            try:
                if hasattr(session.env, "close"):
                    session.env.close()
                del session.env
                del session
            except Exception as e:
                logger.warning(f"Stop failed to clean env: {e}")
        return {"message": "Session stopped."}


@app.post("/reset")
async def reset(req: ResetRequest):
    async with lock:
        session = env_sessions.get(req.session_id)
        if not session:
            logger.info(f"[RESET] session_id: {req.session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            session.env.reset()
            session.last_used = datetime.utcnow()
            return {"message": "Environment reset."}
        except Exception as e:
            logger.exception(f"Reset failed for session {req.session_id}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_attribute")
async def get_attribute(session_id: str, attribute_name: str):
    async with lock:
        logger.info(f"[GET_ATTRIBUTE] try session_id: {session_id}, attribute_name: {attribute_name}")
        session = env_sessions.get(session_id)
        if not session:
            logger.info(f"[GET_ATTRIBUTE] session_id: {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            attr_value = getattr(session.env, attribute_name, None)
            if callable(attr_value):
                attr_value = attr_value()
            session.last_used = datetime.utcnow()
            logger.info(
                f"[GET_ATTRIBUTE] successed session_id: {session_id}, attribute_name: {attribute_name}, attribute_value: {attr_value}"
            )
            return {"attribute_value": attr_value}
        except Exception as e:
            logger.exception(
                f"Get attributes failed for session {session_id}, attribute_name: {attribute_name}, env {session.env}")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import pytest
    from fastapi.testclient import TestClient
    client = TestClient(app)
    env_str = """AddBinary@{"a": "101", "b": "101"}"""
    session_id = "123"

    def test_start():
        response = client.post("/start", json={"session_id": session_id, "env_str": env_str, "env_name": "synth_env"})
        print(response.json())
        print(response.status_code)

    def test_step():
        client.post("/start", json={"session_id": session_id, "env_str": env_str})
        action = '{\n"name":"done",\n"parameters":{\n"answer":"101"\n}\n}'

        response = client.post("/step", json={"session_id": session_id, "action": action})

        data = response.json()
        print(data)
        assert "status" in data
        assert "observation" in data

    def test_is_done():
        client.post("/start", json={"session_id": session_id, "env_str": env_str})
        response = client.get(f"/is_done?session_id={session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "done" in data

    def test_is_success():
        client.post("/start", json={"session_id": session_id, "env_str": env_str})
        response = client.get(f"/is_success?session_id={session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_stop():
        client.post("/start", json={"session_id": session_id, "env_str": env_str})
        response = client.post("/stop", json={"session_id": session_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session stopped."

    def test_reset():
        client.post("/start", json={"session_id": session_id, "env_str": env_str})
        response = client.post("/reset", json={"session_id": session_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Environment reset."

    pytest.main([__file__])
