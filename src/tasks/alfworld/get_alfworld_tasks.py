import yaml
import random
import json
import os
from tqdm import tqdm
from glob import glob
import shutil

from src.utils.environment.alfworld.alfworld.alfworld.agents.environment.alfred_tw_env import (
    AlfredTWEnv,
)

ALFWORLD_ENV_SEED = 38

def get_alfworld_single_agent_tasks():
    """Gets a json file of all of the Alfworld validation tasks."""

    alfworld_config_path = "src/utils/environment/alfworld/alfworld/configs/base_config.yaml"

    with open(alfworld_config_path) as file:
        alfworld_cfg = yaml.safe_load(file)

    split = "eval_out_of_distribution"

    random.seed(ALFWORLD_ENV_SEED)

    c = AlfredTWEnv(alfworld_cfg, train_eval=split)
    c.game_files.sort()
    c = c.init_env(batch_size=1)

    tasks = list()

    for _ in tqdm(range(len(c.gamefiles))):
        ob, info = c.reset()  # type:ignore
        ob = "\n".join(ob[0].split("\n\n")[1:])
        initial_scene_observation, task_natural_language = ob.split("\n")
        task_name, scene = info["extra.gamefile"][0].split("/")[-3:-1]

        tasks.append({
            "scene": scene,
            "task_name": task_name,
            "task_natural_language": task_natural_language
        })

        with open("temp.json", "w") as file:
            json.dump(tasks, file, indent=3)

def get_alfworld_consistency_tasks(sample_size: int = 10):
    alfworld_config_path = "src/utils/environment/alfworld/alfworld/configs/base_config.yaml"

    consistency_set_path = "src/tasks/alfworld/consistency_tasks"
    consistency_set_json_path = "src/tasks/alfworld/alfworld_consistency_tasks.json"
    if (os.path.exists(consistency_set_path) 
        and os.path.exists(consistency_set_json_path) 
        and len(glob(os.path.join(consistency_set_path, "*"))) > 0
    ):
        return

    with open(alfworld_config_path) as file:
        alfworld_cfg = yaml.safe_load(file)

    split = "eval_out_of_distribution"

    random.seed(ALFWORLD_ENV_SEED)

    c = AlfredTWEnv(alfworld_cfg, train_eval=split)
    c.game_files.sort()
    c = c.init_env(batch_size=1)

    tasks = list()

    for idx in tqdm(range(len(c.gamefiles))):
        ob, info = c.reset()  # type:ignore
        ob = "\n".join(ob[0].split("\n\n")[1:])
        initial_scene_observation, task_natural_language = ob.split("\n")
        task_name, scene = info["extra.gamefile"][0].split("/")[-3:-1]

        tasks.append({
            "scene": scene,
            "task_name": task_name,
            "task_natural_language": task_natural_language,
            "idx": idx
        })

    sampled_tasks = random.sample(tasks, sample_size)
    sampled_tasks.sort(key=lambda x: x["idx"])

    # here we will create a directory containing the alfworld game files for the 
    # sampled tasks
    validation_set_path = "src/utils/environment/alfworld/alfworld/alfworld/data/json_2.1.1/valid_unseen"
    consistency_set_path = "src/tasks/alfworld/consistency_tasks"

    # clear out any residual consistency dataset
    if os.path.exists(consistency_set_path):
        shutil.rmtree(consistency_set_path)
    os.makedirs(consistency_set_path, exist_ok=True)

    for task in sampled_tasks:
        # copy the game file to the consistency task directory
        source_task_dir = os.path.join(validation_set_path, task["task_name"], task["scene"])
        target_task_dir = os.path.join(consistency_set_path, task["task_name"], task["scene"])
        shutil.copytree(source_task_dir, target_task_dir)

    # re-load the alfworld environment to get the correct gamefile order
    # that the simulator will load things in
    alfworld_config_path = "src/utils/environment/alfworld/alfworld_consistency_config.yaml"

    with open(alfworld_config_path) as file:
        alfworld_cfg = yaml.safe_load(file)

    c = AlfredTWEnv(alfworld_cfg, train_eval=split)
    c.game_files.sort()
    c = c.init_env(batch_size=1)

    print("Building consistency task json file..")
    tasks = list()
    for idx in tqdm(range(len(c.gamefiles))):
        ob, info = c.reset()  # type:ignore
        ob = "\n".join(ob[0].split("\n\n")[1:])
        initial_scene_observation, task_natural_language = ob.split("\n")
        task_name, scene = info["extra.gamefile"][0].split("/")[-3:-1]

        tasks.append({
            "scene": scene,
            "task_name": task_name,
            "task_natural_language": task_natural_language
        })

    with open("src/tasks/alfworld/alfworld_consistency_tasks.json", "w") as file:
        json.dump(tasks, file, indent=3)
