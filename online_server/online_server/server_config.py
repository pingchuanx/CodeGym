import json, os, re
import importlib.util

# current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# construct target file path
codegym_v1_target_file = os.path.normpath(os.path.join(current_dir, "envs", "codegym_v1"))

# env_name corresponding class path
env_name2path = {
    "codegym_v1": codegym_v1_target_file,
}

# env_name corresponding prefix to file name mapping
env_name2dict = {
    "codegym_v1": {},
}


def extract_prefix_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # try to extract prefix from from_env_str method
    match = re.search(r'def\s+from_env_str\s*\(.*?\):.*?prefix\s*=\s*[\'"]([^\'"]+)[\'"]', code, re.DOTALL)
    if match:
        return match.group(1)
    return None

# create corresponding prefix to file name mapping
# e.g. TauBench@ will be mapped to TauBenchEnv.py, for dynamic loading
def build_prefix_to_file_map_by_code(env_dir):
    prefix_to_file = {}

    for filename in os.listdir(env_dir):
        if filename.endswith("Env.py") and not filename.startswith("__"):
            filepath = os.path.join(env_dir, filename)
            prefix = extract_prefix_from_file(filepath)
            if prefix:
                prefix_to_file[prefix] = filename
            else:
                print(f"⚠️ failed to extract prefix: {filename}")
                raise ValueError(f"failed to extract prefix: {filename}")

    return prefix_to_file

def build_prefix_to_file_map_by_name(env_dir):
    prefix_to_file = {}

    for filename in os.listdir(env_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            prefix_to_file[filename[:-3] + '@'] = filename
    
    return prefix_to_file

for env_name, env_path in env_name2path.items():
    if env_name == "codegym_v1":
        prefix_to_file = build_prefix_to_file_map_by_name(env_path)
    else:
        raise ValueError(f"unknown env name: {env_name}")
    env_name2dict[env_name] = prefix_to_file

# preload some env class
# _preload_env_list = [("tau_bench", "TauBench@")]
_preload_env_list = []
_preload = {}

# cache preloaded class
_cached = {}

def get_class(env_name, prefix):
    if (env_name, prefix) in _preload:
        return _preload[(env_name, prefix)]
    if (env_name, prefix) in _cached:
        return _cached[(env_name, prefix)]
    env_cls_dir = env_name2path[env_name]
    env_cls_name = env_name2dict[env_name][prefix]
    env_cls_path = os.path.join(env_cls_dir, env_cls_name)
    # dynamic import
    spec = importlib.util.spec_from_file_location(env_cls_name, env_cls_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # get class from module
    if env_name == "codegym_v1":
        env_cls = getattr(module, "__".join(env_cls_name.split("__")[1:])[:-3]) # remove .py and split by __
    else:
        env_cls = getattr(module, env_cls_name[:-3]) # remove .py
    # delete other content
    del spec
    del module
    _cached[(env_name, prefix)] = env_cls
    return env_cls

if _preload_env_list:
    for env_name, prefix in _preload_env_list:
        env_cls = get_class(env_name, prefix)
        _preload[(env_name, prefix)] = env_cls

if __name__ == "__main__":
    # tau_bench
    # env_dict = {"env_name": "retail", "user_strategy": "llm", "user_model": "agent_20b_qrl_taubench", "task_split": "train", "user_provider": "groot", "task_index": 0, "user_response": "help me return the item"}
    # env_str = "TauBench@" + json.dumps(env_dict, ensure_ascii=False)
    # env = TauBenchEnv.from_env_str(env_str)
    # print(env)
    # codegym_v1
    cls = get_class("codegym_v1", "Code_Contests_1036_I__MaxUniqueDepthEnv@")
    print(cls)
    env = cls.from_env_str("MaxUniqueDepthEnv@{\"n\": 2, \"E\": 5, \"depths\": [[1, 3], [2, 3]]}")
    print(env.solve())