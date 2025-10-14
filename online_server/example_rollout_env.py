import time
import uuid
import functools
import requests
import os
import json
import threading
from collections import defaultdict

RETRY_TIMES = 3
ERROR_ALLOW_TIMES = 5
def retry_function(exceptions, tries=1000000, delay=2, backoff=2):
    """
    Wrap the function in retry logic.

    :param exceptions: tuple, the exception types to capture
    :param tries: int, maximum number of retries
    :param delay: int, the initial waiting interval (seconds)
    :param backoff: int, the factor by which the waiting time increases for each retry
    :return: the wrapped function
    """

    def decorator_retry(func):

        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"{func.__name__} failed: {e}, Retrying in {_delay} seconds...")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)

        return wrapper_retry

    return decorator_retry


class EnvRemoteInstance:
    uid: str
    port: str

    def __init__(self, uid, port) -> None:
        self.port = port
        self.uid = uid


class EnvManagerClient(object):

    def __init__(self, manager_host):
        self._host = manager_host

    @retry_function(exceptions=(Exception,), tries=1000000, delay=1, backoff=2)
    def get_one_instance(self):
        resp = requests.get(f"{self._host}/get_instance")
        if resp.status_code != 200:
            raise Exception(f"get one instance failed, status code: {resp.status_code}, resp: {resp.text}")
        data = resp.json()
        return EnvRemoteInstance(**data)

    @classmethod
    @retry_function(exceptions=(Exception,), tries=1000000, delay=1, backoff=2)
    def release_instance(cls, host, uid):
        resp = requests.get(f"{host}/release_instance?uid={uid}")
        if resp.status_code != 200:
            raise Exception(f"get one instance failed, status code: {resp.status_code}, resp: {resp.text}")
        return


class EnvClient(object):

    def __init__(self, host, session, env_str):
        self._host = host
        self._session = session
        self._env_str = env_str

    # @retry_function(exceptions=(Exception,), tries=1000000, delay=2, backoff=2)
    def start(self, env_name="synth_env"):
        num = 10
        data = None
        for i in range(num):
            try:
                resp = requests.post(f"{self._host}/start",
                                    json={
                                        "session_id": self._session,
                                        "env_str": self._env_str,
                                        "env_name": env_name
                                    },
                                    timeout=120)
                try:
                    data = resp.json()
                    obs = data["observation"]
                except:
                    obs = "env start successfully"
                assert "[ERROR after 3 attempts]" not in obs
                resp.raise_for_status()  # automatically check the status code, if not 2xx, throw HTTPError
                return obs
            except Exception as e:
                message = str(e)
        print(f"[Start] Fail after {num} trials | {env_name} | session: {self._session}, env str: {self._env_str}, data: {data}, error: {message}")

    def step(self, action, timeout, max_retries=RETRY_TIMES):
        data = None
        message = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(f"{self._host}/step",
                                    json={
                                        "session_id": self._session,
                                        "action": action,
                                        "timeout": timeout
                                    },
                                    timeout=timeout)
                data = resp.json()  # parse JSON
                resp.raise_for_status()  # automatically check the status code, if not 2xx, throw HTTPError
                assert "[ERROR after 3 attempts]" not in data["observation"] and "[generate_next_message error" not in data["observation"]
                return data["status"], data["observation"]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                message = str(e)
                
        print(f"[Step] Fail after {max_retries} trials | session: {self._session}, env str: {self._env_str}, action: {action}, content: {data}, error: {message}")
        return True, f'Error occurred: {message}'

    def is_done(self, timeout=30):
        try:
            resp = requests.get(f"{self._host}/is_done?session_id={self._session}", timeout=timeout)
            resp.raise_for_status()  # if the status code is not 2xx, throw exception, and do not execute the following code
            return resp.json()["done"]
        except Exception as e:
            print("[IS_DONE] HTTP error occurred:", e)
            return True

    def reward(self, timeout=30):
        try:
            resp = requests.get(f"{self._host}/reward?session_id={self._session}", timeout=timeout)
            resp.raise_for_status()  # if the status code is not 2xx, throw exception, and do not execute the following code
            return resp.json()["reward"]
        except Exception as e:
            print("[REWARD] HTTP error occurred:", e)
            return 0

    def is_success(self, timeout=30):
        try:
            resp = requests.get(f"{self._host}/is_success?session_id={self._session}", timeout=timeout)
            if resp.status_code != 200:
                raise Exception(f"get is_success failed, status code: {resp.status_code}, resp: {resp.text}")
        except requests.exceptions.Timeout:
            print(f"[{self._host}/is_success] session: {self._session} is_success timeout, env str: {self._env_str}")
        except requests.exceptions.RequestException as e:
            print(
                f"[{self._host}/is_success] session: {self._session} is_success failed, {e}, env str: {self._env_str}")
        return resp.json()['success']

    def get_attrubute(self, attribute_name, timeout=30):
        try:
            resp = requests.get(
                f"{self._host}/get_attribute?session_id={self._session}&attribute_name={attribute_name}", timeout=timeout)
            resp.raise_for_status()  # if the status code is not 2xx, throw exception, and do not execute the following code
            return resp.json()["attribute_value"]
        except Exception as e:
            print("[GET_ATTRIBUTE] HTTP error occurred:", e)
            return None


class OnlineGymEnv:
    '''
    for GymEnv pugin
    '''

    def __init__(self, ip, env_str, **kwargs):
        self.session_id = str(uuid.uuid4())
        self.ip = ip
        self.env_manager_client = EnvManagerClient(self.ip + ":8000")
        self.env_manager_instance = self.env_manager_client.get_one_instance()
        self.env_host = self.ip + ":" + str(self.env_manager_instance.port)
        print(f"[GymEnv Plugin] start init env at {self.env_host}")

        self.env_client = EnvClient(self.env_host, self.session_id, env_str)
        self.initial_obs = self.env_client.start(env_name=kwargs.get("env_name", "synth_env"))
        self.action_list = None
        self.pre_append_code = None
        print(f"[GymEnv Plugin] env init succeed {env_str} at {self.env_host}")

    @classmethod
    def from_env_str(cls, env_str: str, max_turns=100):
        prefix = env_str.split("@", 1)[0]
        if prefix == 'codegym_v1':
            SERVER_IP_ADDRESS = os.environ.get("SERVER_IP_ADDRESS", "::")
            env_name="codegym_v1"
        else:
            raise Exception(f"env_str_prefix {prefix} not supported")
        assert SERVER_IP_ADDRESS, "SERVER_IP_ADDRESS is not set"
        env_str = env_str.split("@", 1)[1]
        gym = cls(ip=f"http://[{SERVER_IP_ADDRESS}]", env_str=env_str, env_name=env_name)
        return gym

    def step(self, action, timeout=None):
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        return self.env_client.is_done()

    @property
    def reward(self):
        return int(self.env_client.reward() > 0)

    def release(self):
        print(f"[GymEnv Plugin] release instance: {self.ip}:8000 {self.env_manager_instance.uid}")
        EnvManagerClient.release_instance(
            self.ip + ":8000", self.env_manager_instance.uid
        )
        # send /stop request to env_host
        try:
            stop_url = f"{self.env_host}/stop"
            resp = requests.post(
                stop_url, json={"session_id": self.session_id}, timeout=10
            )
            print(f"[GymEnv Plugin] Stop response: {resp.status_code} {resp.text}")
        except Exception as e:
            print(
                f"[GymEnv Plugin] Failed to send stop request to {self.env_host}: {e}"
            )

    def __del__(self):
        pass

class OnlineFcGymEnv(OnlineGymEnv):
    env_str_prefix = "OnlineFcGymEnv"
    error_count = 0

    count_step = 0 
    _count_step_lock = threading.Lock()

    # action_stat: used for compute the success number of actions
    # structure: { "action_name": {"success": N, "failure": M} }
    action_stat = defaultdict(lambda: {"success": 0, "failure": 0})
    _action_stat_lock = threading.Lock()

    def step(self, action, timeout=None):
        FC_TIME_OUT = 10
        if self.error_count > ERROR_ALLOW_TIMES: # error too many times, set timeout to 1s
            FC_TIME_OUT = 1
        action = action.strip()
        try:
            json.loads(action)
        except Exception as e:
            return False, f"The action cannot be parsed in json format {action}, {e}"
        if len(action) > 0 and action[0] == "[" and action[-1] == "]":
            action = action[1:-1]
        status, resp = self.env_client.step(action, FC_TIME_OUT)
        success = False if 'Error occur' in resp else True
        if not success:
            self.error_count += 1
            print(f"[OnlineFcGymEnv] step error {self.error_count} times on a row, resp: {resp}")
        else:
            self.error_count = 0
        self.update_action_stat(action, success)
        return status, resp
    
    def update_action_stat(self, action, success):
        try:
            call_dict = json.loads(action)
            action_name = call_dict["name"]
        except:
            action_name = "unknow"
        with self._count_step_lock:
            OnlineFcGymEnv.count_step += 1
        with self._action_stat_lock:
            if success:
                self.action_stat[action_name]["success"] += 1
            else:
                self.action_stat[action_name]["failure"] += 1
        if OnlineFcGymEnv.count_step % 1000 == 0:
            pass
            # print(f"\n--- [OnlineFcGymEnv] Current Step Count: {OnlineFcGymEnv.count_step} ---")
            # print full action_stat (optional, if the data is too large, it will be very long)
            # print(f"Full Action Stat: {dict(OnlineFcGymEnv.action_stat)}") 

    def print_top_failed_actions(self, top_n=20, min_calls=5):
        """
        print the actions with the lowest success rate.
        min_calls: the minimum number of times the action must be executed to be included in the statistics.
        """
        failed_actions = []
        for action_name, stats in OnlineFcGymEnv.action_stat.items():
            success_count = stats["success"]
            failure_count = stats["failure"]
            total_count = success_count + failure_count

            if total_count >= min_calls: # only count the actions that have been executed at least min_calls times
                if total_count > 0:
                    success_rate = (success_count / total_count) * 100
                else: # this should not happen, because total_count >= min_calls
                    success_rate = 0.0 
                
                failed_actions.append({
                    "action_name": action_name,
                    "success_rate": success_rate,
                    "total_calls": total_count,
                    "success_count": success_count,
                    "failure_count": failure_count
                })
        
        # sort by success rate
        failed_actions.sort(key=lambda x: x["success_rate"])

        print(f"Top {top_n} Actions with Lowest Success Rate (min_calls >= {min_calls}):")
        if not failed_actions:
            print("  No actions meet the criteria yet.")
            return

        for i, action_info in enumerate(failed_actions[:top_n]):
            print(f"{i+1}. Action: '{action_info['action_name']}' | "
                  f"Success Rate: {action_info['success_rate']:.2f}% | "
                  f"Total Calls: {action_info['total_calls']} "
                  f"(Success: {action_info['success_count']}, Failure: {action_info['failure_count']})")

        print("-" * 50) # separator

class codegym_v1(OnlineFcGymEnv):
    env_str_prefix = "codegym_v1"

if __name__ == '__main__':
    env_str = "codegym_v1@Code_Contests_14164_I__MaxRotationSumEnv@{\"array\": [-999999999, -999999999, -999999999, -999999999, -999999999]}"
    example_env = codegym_v1.from_env_str(env_str)
    print(example_env.step('{"name":"Done", "parameters":{"answer":-4999999995}}'))
    print(example_env.reward)
    example_env.release()