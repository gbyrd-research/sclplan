import os
import random
from typing import List

import yaml
import json
from tqdm import tqdm
import random

from src.tasks.base import TaskMetadata, Tasks
from src.utils.environment.alfworld.alfworld.alfworld.agents.environment.alfred_tw_env import (
    AlfredTWEnv,
)
from src.utils.environment.alfworld.env import ALFWORLD_ENV_SEED


class AlfworldTaskMetadata(TaskMetadata):
    def __init__(
        self,
        scene: str,
        task_name: str,
        task_natural_language: str,
    ):
        super().__init__(scene, task_name, task_natural_language)


class AlfworldTasks(Tasks):
    def __init__(self, task_type: str, num_tasks: int = 50):
        self.num_tasks = num_tasks
        self.tasks = self._get_tasks(task_type)

    def _get_tasks(self, task_type: str) -> List[AlfworldTaskMetadata]:
        # create an alfworld environment
        if task_type == "single_agent" or task_type == "consistency":

            tasks_json_file = os.path.join(
                "src", "tasks", "alfworld", f"alfworld_{task_type}_tasks.json"
            )

            if not os.path.exists(tasks_json_file):
                # load the alfworld config
                config_path = os.path.join(
                    "src", "utils", "environment", "alfworld", "alfworld_base_config.yaml"
                )
                with open(config_path) as file:
                    config = yaml.safe_load(file)

                # Set seed for reproducability
                random.seed(ALFWORLD_ENV_SEED)

                env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
                env.game_files = random.sample(env.game_files, self.num_tasks)
                env.game_files.sort()
                env = env.init_env(batch_size=1)

                NUM_GAMEFILE = len(env.gamefiles)

                tasks = list()

                for n in tqdm(range(NUM_GAMEFILE)):

                    ob, info = env.reset()
                    ob = "\n".join(ob[0].split("\n\n")[1:])
                    _, task_natural_language = ob.split("\n")
                    task_name, scene = info["extra.gamefile"][0].split("/")[-3:-1]

                    task_dict = {
                        "scene": scene,
                        "task_name": task_name,
                        "task_natural_language": task_natural_language
                    }

                    tasks.append(task_dict)

                with open(tasks_json_file, "w") as file:
                    json.dump(tasks, file, indent=3)

            with open(tasks_json_file, "r") as file:
                tasks_list = json.load(file)

            tasks = list()
            for idx, task in enumerate(tasks_list):
                tasks.append(AlfworldTaskMetadata(
                    scene=task["scene"],
                    task_name=f"Alfworld_Validation_Task_{str(idx).zfill(3)}",
                    task_natural_language=task["task_natural_language"]
                ))

            sampled_tasks = random.sample(tasks, self.num_tasks)

            ordered_sampled_tasks = [task for task in tasks if task in sampled_tasks]

            return ordered_sampled_tasks
        
        else:
            raise NotImplementedError(f"Task type {task_type} not yet implemented.")
        
    def __iter__(self):  # type:ignore
        self.index = 0
        return self

    def __next__(self):  # type:ignore
        if self.index < len(self.tasks):
            value = self.tasks[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration
