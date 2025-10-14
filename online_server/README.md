# Online Server for CodeGym

A distributed environment server for running CodeGym programming environments at scale.

## Overview

The Online Server provides a scalable solution for running CodeGym programming environments. It manages multiple environment instances to support distributed training and validation workloads.

## Setup

### Step 1: Install required packages
```bash
conda create -n online_server python==3.11.2
conda activate online_server
pip3 install -r requirements.txt
```

### Step 2: Download CodeGym Environments

Download the CodeGym environments from [HuggingFace](https://huggingface.co/datasets/VanishD/CodeGym) and extract them into the `online_server/online_server` directory.

**Expected directory structure:**
```
online_server/
└── online_server/
    └── envs/
        └── codegym_v1/
            ├── [envs].py
            └── ...
```

## Configuration

### Worker Requirements

The number of workers must be sufficient to handle your training and validation workloads:

**Formula:** `num-workers > training_instances + validation_instances`

**Example calculation (GRPO algorithm):**
- Training batch size: 32
- Number of sample per training instance: 8  
- Validation batch size: 128
- **Required workers:** > 32 × 8 + 128 = **384 workers**

## Usage

### Starting the Environment Server

```bash
python3 online_server/env_instance_manager.py --workers [num-workers]
```

**Parameters:**
- `--workers`: Number of worker processes to spawn (must exceed your workload requirements)

### Testing

After launching the environment server, you can test the connection:

```bash
# Test endpoint (replace with actual endpoint when available)
curl "http://0.0.0.0:8000/get_instance"
# Release endpoint
curl "http://[::]:8000/release_instance?uid=[uid returned in last step]"
```

### Linking a Worker to the Server

The file `example_rollout_env.py` provides an example implementation of how to connect a worker to the server on the rollout side. It illustrates the connection design, but you may need to adapt or extend it to fit your own training framework.

The environment interface includes the following methods:

**1. `from_env_str(env_str: str)`**

Initializes an environment from a string specification. The format is:

```
[env_class]@[env_name]@[env_task_config]
```

* **`env_class`**: The class of the environment (default: `codegym_v1`).
* **`env_name`**: The specific environment name, e.g. `Code_Contests_14164_I__MaxRotationSumEnv`.
* **`env_task_config`**: A JSON-formatted dictionary specifying task configuration.
  Example:

  ```json
  {"array": [-999999999, -999999999, -999999999, -999999999, -999999999]}
  ```

**2. `step(action_str: str, timeout: Optional[int] = None) -> str`**

Executes one step in the environment given an action.

* **Parameters**:

  * `action_str`: The action to execute (string format).
  * `timeout`: (Optional) Maximum allowed time for the step execution.

* **Returns**:

  * `obs_str`: A string representing the observation after the action.

**3. `finished() -> bool`**

Checks whether the current episode has terminated.

* **Returns**:

  * `True` if the episode has ended.
  * `False` otherwise.

**4. `reward() -> int`**

Retrieves the reward for the current episode.

* **Returns**:

  * An integer reward value. By default, the reward is binary (0 or 1).