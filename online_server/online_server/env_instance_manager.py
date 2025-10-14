import os
import uuid
import time
import socket
import subprocess
import threading
from datetime import datetime
from fastapi import FastAPI
from typing import List
import heapq
import psutil
import atexit
import requests
from contextlib import asynccontextmanager

CLEANUP_INTERVAL_SECONDS = 600  # check every 10 min
INSTANCE_TTL_SECONDS = 3600 * 3 # instance ttl is 3 hours


def log(msg):
    print(f"[{datetime.now()}] {msg}")


# -------------- Port Monitor --------------
class PortMonitor:
    _lock = threading.Lock()

    @classmethod
    def get_available_ports(cls, start_port: int, count: int) -> List[int]:
        with cls._lock:
            ports = []
            port = start_port
            while len(ports) < count and port < 65535:
                if cls.is_port_available(port):
                    ports.append(port)
                port += 1
            if len(ports) < count:
                raise Exception("Not enough available ports.")
            return ports

    @staticmethod
    def is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        return not PortMonitor.is_port_available(port)

    @classmethod
    def start(cls, cmd: List[str], cwd: str, port: int, log_file: str):

        def monitor_log_size():
            while True:
                if os.path.exists(log_file):
                    file_size = os.path.getsize(log_file)
                    if file_size > 10 * 1024 * 1024:  # 10MB
                        with open(log_file, 'w') as f:
                            f.write('')  # clear log file
                time.sleep(CLEANUP_INTERVAL_SECONDS)

        # start log monitor thread
        log_monitor_thread = threading.Thread(target=monitor_log_size)
        log_monitor_thread.daemon = True
        log_monitor_thread.start()

        with open(log_file, 'a') as file:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=file, stderr=file)
        return proc


# -------------- Instance Class --------------
class EnvInstance:

    def __init__(self, cwd: str, port: int, logfile_dir: str):
        self.uid = str(uuid.uuid4())
        self.port = port
        self.cwd = cwd
        self.proc = None
        self.logfile_dir = logfile_dir
        os.makedirs(logfile_dir, exist_ok=True)
        self.start_time = time.time()
        self.last_used = time.time()
        self.start()

    def start(self):
        self.start_time = time.time()
        log_file = os.path.join(self.logfile_dir, f'env_{self.port}.log')
        self.proc = PortMonitor.start(
            cmd=["uvicorn", "env_server:app", "--port",
                 str(self.port), "--host", "::", "--workers", "1"],
            cwd=self.cwd,
            port=self.port,
            log_file=log_file)

    def stop(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()

    def get_url(self):
        host = os.environ.get("SERVER_PUBLIC_HOST")
        if host:
            return f"http://{host}:{self.port}"
        try:
            response = requests.get("https://httpbin.org/ip", timeout=3)
            ip = response.json()["origin"].split(",")[0].strip()
            return f"http://{ip}:{self.port}"
        except Exception as e:
            print(f"[WARN] failed to get public IP: {e}")
            return f"http://127.0.0.1:{self.port}"

    def get_port(self):
        return self.port

    def health_check(self):
        return self.proc.poll() is None

    def __lt__(self, other):  # smaller <
        return True

    def __gt__(self, other):  # larger >
        return True

    def __eq__(self, other):  # equal ==
        return True


# -------------- Priority Queue --------------
class PriorityQueue:

    def __init__(self):
        self.heap = []

    def push(self, item, timestamp):
        heapq.heappush(self.heap, (timestamp, item))

    def pop(self):
        if not self.heap:
            raise IndexError("PriorityQueue is empty")
        return heapq.heappop(self.heap)[1]


# -------------- Instance Manager --------------
class EnvInstanceManager:
    _lock = threading.Lock()
    _log_root = "logs"
    _cwd = os.getcwd()
    _all_instances = {}
    _free_instances = PriorityQueue()
    _allocated_instances = {}

    @classmethod
    def init_instances(cls, count=3):
        ports = PortMonitor.get_available_ports(10000, count)
        for port in ports:
            inst = EnvInstance(cls._cwd, port, cls._log_root)
            cls._all_instances[inst.uid] = inst
            cls._free_instances.push(inst, inst.start_time)
        threading.Thread(target=cls._cleanup_loop, daemon=True).start()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if not cls._free_instances.heap:
                log("❌ No available resources to allocate.")
                raise Exception("❌ No available resources to allocate.")
            while True:
                inst = cls._free_instances.pop()
                if inst.health_check():
                    inst.last_used = time.time()
                    cls._allocated_instances[inst.uid] = inst
                    cls.debug()
                    return inst
                else:
                    cls.debug()
                    inst.stop()
                    inst.start()
                    cls._free_instances.push(inst, inst.start_time)

    @classmethod
    def release_instance(cls, uid):
        with cls._lock:
            inst = cls._allocated_instances.pop(uid, None)
            if inst:
                inst.last_used = time.time()
                cls._free_instances.push(inst, time.time())
                cls.debug()

    @classmethod
    def cleanup(cls):
        with cls._lock:
            for inst in list(cls._all_instances.values()):
                inst.stop()
            cls._all_instances.clear()
            cls._allocated_instances.clear()
            cls._free_instances = PriorityQueue()
            cls.debug()

    @classmethod
    def _cleanup_loop(cls):
        while True:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            now = time.time()
            with cls._lock:
                for uid, inst in list(cls._allocated_instances.items()):
                    if now - inst.last_used > INSTANCE_TTL_SECONDS:
                        log(f"[CLEANUP] instance {uid} expired, auto-released to free pool (port {inst.port}).")
                        cls._allocated_instances.pop(uid, None)
                        inst.last_used = now
                        cls._free_instances.push(inst, now)
                cls.debug()

    @classmethod
    def debug(cls):
        log(f"[{datetime.now()}] all_instances count: {len(cls._all_instances)}, allocated_instances count: {len(cls._allocated_instances)}, free_instances count: {len(cls._free_instances.heap)}"
           )


# -------------- FastAPI App --------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # when starting
    workers = app.state.workers
    EnvInstanceManager.init_instances(count=workers)
    print("EnvInstanceManager started.")
    yield
    # when closing
    EnvInstanceManager.cleanup()
    print("EnvInstanceManager stopped and all ports released.")


app = FastAPI(lifespan=lifespan)


@app.get("/get_instance")
def get_instance():
    inst = EnvInstanceManager.get_instance()
    return {"uid": inst.uid, "port": inst.get_port()}


@app.get("/release_instance")
def release_instance(uid: str):
    EnvInstanceManager.release_instance(uid)
    return {"uid": uid, "status": "released"}


atexit.register(EnvInstanceManager.cleanup)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--workers', type=int, default=1024)
    args = parser.parse_args()
    app.state.workers = args.workers

    import uvicorn
    uvicorn.run(app, host="::", port=8000)
