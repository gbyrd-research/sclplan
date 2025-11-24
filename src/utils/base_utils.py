import importlib

from omegaconf import DictConfig
from typing import List

from src.agent.single_agent.base import BaseAgent
from src.tasks.ai2thor.tasks import Tasks
from src.utils.environment.base import Environment


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def get_agent(agent_cfg: DictConfig, agent_id: int = 0) -> BaseAgent:
    target = agent_cfg.target
    params = agent_cfg.params
    return get_obj_from_str(target)(**params, agent_id=agent_id)


def get_multi_agent(cfg: DictConfig, agent_count: int):
    if cfg.multi_agent.name == "master_planner_agent":
        cfg_MP = cfg.multi_agent.cfg
        cfg_BA = cfg.agent.params.agent_cfg
        cfg_AI2THOR_TE = cfg.ai2thor_tool_engine
        target = cfg.multi_agent.target
        params = {
            "cfg_MP": cfg_MP,
            "cfg_BA": cfg_BA,
            "cfg_AI2THOR_TE": cfg_AI2THOR_TE,
            "agent_count": agent_count,
        }
        return get_obj_from_str(target)(**params)
    else:
        raise NotImplementedError("Not implemented yet.")


def get_tasks(task_cfg: DictConfig) -> Tasks:
    target = task_cfg.target
    params = task_cfg.params
    return get_obj_from_str(target)(**params)


def get_env(target: str) -> Environment:
    env_class: Environment = get_obj_from_str(target)

    # throw an exception if the initialize method has not been defined
    # for the environment class
    if not (hasattr(env_class, "initialize") 
            and callable(getattr(env_class, "initialize", None))):
        raise AttributeError("Environment class does not have required method 'initialize'.")

    return env_class
