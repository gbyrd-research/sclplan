import asyncio
import logging
import os
import traceback

import hydra
from omegaconf import DictConfig

from src.agent.single_agent.base import BaseAgent
from src.tasks.base import Tasks
from src.utils.analysis.process_results import process_results
from src.utils.base_utils import get_agent, get_env, get_tasks
from src.utils.misc.logging_utils import set_logging_thresholds
from src.utils.environment.base import Environment
from src.tasks.alfworld.get_alfworld_tasks import get_alfworld_consistency_tasks


def run(env: Environment, agent: BaseAgent, tasks: Tasks):
    for idx, task_metadata in enumerate(tasks):
        env.reset(task_metadata)
        task_metadata.print()
        task_result_save_path = os.path.join(
            agent.results_dir, task_metadata.task_name, "results.json"
        )
        if os.path.exists(task_result_save_path):
            print("Results already generated for task. Skipping...")
            continue
        try:
            result = asyncio.run(agent.perform_task(task_metadata))
        except Exception as e:
            with open(os.path.join(agent.logging_dir, "exception.txt"), "a") as file:
                file.write(traceback.format_exc())
        # result = asyncio.run(agent.perform_task(task_metadata))
        agent.save_results(task_metadata)
    process_results(agent.results_dir)


@hydra.main(
    version_base=None, config_path="hydra_conf/", config_name="single_agent_default"
)
def main(cfg: DictConfig):
    valid_experiment_types = {"general", "consistency"}
    if cfg.experiment_type not in valid_experiment_types:
        raise Exception(f"Invalid experiment type {cfg.experiment_type}. Must be one of {', '.join(valid_experiment_types)}.")
    
    if cfg.experiment_type == "general":
        num_runs = 1
    elif cfg.experiment_type == "consistency":
        if cfg.env.name != "alfworld_consistency":
            raise Exception("Make sure that you are running the alfworld consistency environment config!")
        # make sure the consistency tasks have been sampled
        get_alfworld_consistency_tasks()
        consistency_run_idx = cfg.consistency_run_idx
        if consistency_run_idx is None:
            raise Exception("Must define a run idx if you are running a consistency experiment.")
        # change the alfworld config to only have the sampled tasks

    
    ENV = get_env(cfg.env.target)
    ENV.initialize(cfg.env.cfg, agent_count=1)
    tasks = get_tasks(cfg.tasks)
    agent = get_agent(cfg.agent)
    if cfg.experiment_type == "consistency":
        agent.results_dir = os.path.join(agent.results_dir, f"{consistency_run_idx}")
    ENV.initialize_agents([agent])
    run(ENV, agent, tasks)

if __name__ == "__main__":
    set_logging_thresholds()
    main()
