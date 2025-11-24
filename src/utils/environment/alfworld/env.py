import asyncio
import math
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import yaml
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.server import Event, MultiAgentEvent
from omegaconf import DictConfig
from scipy.spatial import distance

from src.agent.single_agent.base import BaseAgent
from src.tasks.base import TaskMetadata
from src.utils.environment.base import Environment
from src.utils.misc.logging_utils import get_hydra_run_dir
from src.utils.state.agent_state import AgentState
import json

asyncio_lock = asyncio.Lock()

from src.utils.environment.alfworld.alfworld.alfworld.agents.environment.alfred_tw_env import (
    AlfredTWEnv,
)

with open("src/utils/environment/alfworld/alfworld/configs/base_config.yaml") as file:
    BASE_ALFWORLD_CONFIG = yaml.safe_load(file)

ALFWORLD_ENV_SEED = 42


class AlfworldEnv(Environment):

    is_controller_initialized: bool = False
    is_agent_states_initialized: bool = False
    # 'c' is meant to stand for 'controller'
    c: AlfredTWEnv = None
    cfg: DictConfig = DictConfig({})  # class config
    scene: str = None  # type: ignore
    agent_count: int = 1
    agent_states: Dict = dict()  # type: ignore
    # TODO: Maybe remove the below
    global_task_metadata: TaskMetadata = None  # type:ignore
    num_tasks: int = None
    initial_scene_observation: str = ""

    def __init__(self):
        if not self.is_controller_initialized:
            raise ValueError(
                "Must call AI2ThorEnv.initialize_ai2thor_controller(cfg, agent_count) before attempting to instantiate an object. See class docstring for more information."
            )

    @classmethod
    def initialize(cls, cfg: DictConfig, agent_count: int) -> None:  # type:ignore
        with open(cfg.alfworld_config_path) as file:
            alfworld_cfg = yaml.safe_load(file)

        cls.cfg = cfg
        if cls.cfg.num_tasks is None:
            cls.num_tasks = 134
        else:
            cls.num_tasks = cls.cfg.num_tasks

        split = "eval_out_of_distribution"

        random.seed(ALFWORLD_ENV_SEED)

        cls.c = AlfredTWEnv(alfworld_cfg, train_eval=split)
        # I have found that when running alfworld on different machines (even
        # when setting the seed) the order of the gamefiles of the environment is
        # inconsistent. so, we sort the game files below to keep everything
        # consistent across devices. this is VERY IMPORTANT
        # cls.c.game_files = random.sample(cls.c.game_files, cls.num_tasks)
        cls.c.game_files.sort()

        cls.c = cls.c.init_env(batch_size=1)

        cls.is_controller_initialized = True

        cls.task_complete = False

    @classmethod
    def initialize_agents(cls, agents: List[BaseAgent]) -> None:  # type:ignore
        """Initialize each agent."""
        if len(agents) > 1:
            raise NotImplementedError(
                "Alfworld is not compatible with multiple agents."
            )
        # TODO: flesh this out..
        agent_ids = [a.agent_id for a in agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise Exception("Duplicate agent id found!")
        for agent in agents:
            cls.agent_states[agent.agent_id] = agent.state
            cls.agents[agent.agent_id] = agent

        cls.agent_count = len(agents)
        cls.is_agent_states_initialized = True

    @classmethod
    def reset(cls, global_task_metadata: TaskMetadata):  # type:ignore

        if not cls.is_agent_states_initialized:
            raise ValueError(
                "Must call cls.initialize_agents(agents) before attempting to instantiate an object. See class docstring for more information."
            )

        if not cls.is_controller_initialized:
            raise ValueError(
                "Error: Must call cls.initialize_ai2thor_controller(cfg, agent_count) before attempting to instantiate an object. See class docstring for more information."
            )

        cls.env_steps: list = list()

        ob, info = cls.c.reset()  # type:ignore
        ob = "\n".join(ob[0].split("\n\n")[1:])
        initial_scene_observation, task_natural_language = ob.split("\n")
        task_name, scene = info["extra.gamefile"][0].split("/")[-3:-1]

        cls.initial_scene_observation = initial_scene_observation
        cls.task_natural_language = task_natural_language
        
        cls.scene = scene
        cls.global_task_metadata = global_task_metadata

        # the alfworld environment resets in a set order. we are using a random
        # sample of the environment. the random sample is ordered corresponding
        # to the order of the entire alfworld task validation set, thus, if our
        # alfworld environment does not meet our task description, we will iteratively
        # reset the environment until we get to the alfworld environment that corresponds
        # to our task
        while (
            scene != global_task_metadata.scene
            # or task_name != global_task_metadata.task_name
            or task_natural_language != global_task_metadata.task_natural_language
        ):
            ob, info = cls.c.reset()  # type:ignore
            ob = "\n".join(ob[0].split("\n\n")[1:])
            initial_scene_observation, task_natural_language = ob.split("\n")
            task_name, scene = info["extra.gamefile"][0].split("/")[-3:-1]

            cls.initial_scene_observation = initial_scene_observation
            cls.task_natural_language = task_natural_language

            cls.scene = scene
            cls.global_task_metadata = global_task_metadata

        # # ensure that the environment scene information matches that in
        # # the task metadata
        # if (
        #     scene != global_task_metadata.scene
        #     or task_name != global_task_metadata.task_name
        #     or task_natural_language != global_task_metadata.task_natural_language
        # ):
        #     raise Exception(
        #         "The Alfworld environment scene information does not match that of the passed Task Metadata!"
        #     )

        # reset the objects
        cls.obj_db.reset()

        # get the starting environment observation state
        starting_obj_mdata_list = cls._get_starting_obj_env_mdata_list(
            initial_scene_observation
        )

        # add the starting object metadata to the object database
        [cls.obj_db.add_obj(x) for x in starting_obj_mdata_list]  # so pythonic :)

        # reset agent states and set current location to random object
        for _, a_state in cls.agent_states.items():
            a_state.reset()
            a_state.cur_loc = random.choice(list(cls.obj_db.llm_ids))

        # reset the task complete state to false
        cls.task_complete = False

    @classmethod
    def step(cls, agent_id: int, **kwargs):
        """Takes a step in the Alfworld environment. The agent id
        argument is included for base Environment class consistency,
        but since alfworld is only single agent, it is not used."""
        alfworld_action = kwargs.get("alfworld_action")
        if alfworld_action is None:
            raise Exception("A valid action should always be passed to this function.")

        if len(alfworld_action) != 1:
            raise Exception(
                "We have only implemented single action passing to the alfworld environment."
            )

        # alfworld requires objects in the format object int_id. this makes it
        # difficult to separate the action argument using .split(" "), so we make
        # the objects specified in the format object-int_id so we can easily split
        # we need to process this to remove the "-" before we pass to the
        # command to the environment, so we do this here
        processed_alfworld_action = list()
        for action in alfworld_action:
            if "examineinlight" in action:
                args = action.split(" ")[1:]
                light_source = args[1]
                action = f"use {light_source}"
            processed_alfworld_action.append(action.replace("-", " ").replace("_", " "))

        observation, reward, done, info = cls.c.step(processed_alfworld_action)

        # sometimes move action doesn't trigger observation (e.g. if already close)
        # this is a flaw with Alfworld, so we manually trigger an observation
        if observation[0] == "Nothing happens." and "go to" in alfworld_action[0]:
            observation = (f"You arrive at the {alfworld_action[0].split(' ')[-1]}.",)

        cls.task_complete = done[0]

        observation = observation[0]

        if "Nothing happens" in observation:
            return observation, reward, done, info

        if observation == "":
            scene_changed = True

        scene_obs = {}
        scene_changed = False

        # use last action to update scene_objects
        action_args = cls.parse_action_str_to_tuple(alfworld_action[0])
        action_name = action_args[0]
        action_args = action_args[1:]

        match action_name:
            case "examineinlight":
                obj_id = action_args[0]
                updated_value = action_args[1]
                cls.obj_db.update_obj_mdata(
                    obj_id, "examined", cls.obj_db.ensure_llm_id(updated_value)
                )
            case "go to":
                for obj_id in cls.obj_db.env_ids:
                    cls.obj_db.update_obj_mdata(obj_id, "atReceptacleLocation", False)
                cls.obj_db.update_obj_mdata(
                    action_args[0], "atReceptacleLocation", True
                )
                scene_obs = cls.process_obs(observation)
                scene_changed = len(scene_obs) > 0
            case "open":
                cls.obj_db.update_obj_mdata(action_args[0], "opened", True)
                scene_obs = cls.process_obs(observation)
                scene_changed = len(scene_obs) > 0
            case "close":
                cls.obj_db.update_obj_mdata(action_args[0], "opened", False)
            case "take":
                cls.obj_db.update_obj_mdata(action_args[0], "inReceptacle", False)
                cls.obj_db.update_obj_mdata(action_args[0], "holds", True)
            case "move":
                cls.obj_db.update_obj_mdata(action_args[0], "holds", False)
                cls.obj_db.update_obj_mdata(
                    action_args[0], "inReceptacle", action_args[0], action_args[2]
                )
            case "cool":
                cls.obj_db.update_obj_mdata(action_args[0], "isCool", True)
                cls.obj_db.update_obj_mdata(action_args[0], "isHot", False)
            case "heat":
                cls.obj_db.update_obj_mdata(action_args[0], "isHot", True)
                cls.obj_db.update_obj_mdata(action_args[0], "isCool", False)
            case "clean":
                cls.obj_db.update_obj_mdata(action_args[0], "isClean", True)

        # use observation to update scene_objects
        for receptacle, seen_objects in scene_obs.items():
            cls.obj_db.update_obj_mdata(receptacle, "searched", True)
            receptacle = cls.obj_db.ensure_env_id(receptacle)
            # if you can see objects in receptacle, it must be opened
            if "openable" in cls.obj_db.obj_metadatas[receptacle]:
                cls.obj_db.update_obj_mdata(receptacle, "opened", True)

            # update inReceptacle for all objects observed at this receptacle
            for obj in seen_objects:
                if "-" not in obj:
                    raise Exception(
                        "I expect the object type and integer id to be separated with '-'."
                    )
                if obj not in cls.obj_db.env_ids:
                    obj_mdata = {"objectId": obj, "objectType": obj.split("-")[0]}
                    cls.obj_db.add_obj(obj_mdata)
                obj = cls.obj_db.ensure_env_id(obj)
                llm_obj_id = cls.obj_db.ensure_llm_id(obj)
                llm_receptacle = cls.obj_db.ensure_llm_id(receptacle)
                cls.obj_db.update_obj_mdata(
                    obj, "inReceptacle", llm_obj_id, llm_receptacle
                )
                if "lamp" in obj:
                    cls.obj_db.update_obj_mdata(obj, "isLight", True)

        cls.env_steps.append(kwargs)  # increment the number of steps taken in the environment
        return observation, reward, done, info

    @classmethod
    def parse_action_str_to_tuple(cls, action: str) -> Tuple:
        # pull out the action name
        if "examineinlight" in action:
            action_name = "examineinlight"
        elif "use" in action:
            action_name = "use"
        elif "go to" in action:
            action_name = "go to"
        elif "open" in action:
            action_name = "open"
        elif "close" in action:
            action_name = "close"
        elif "take" in action:
            action_name = "take"
            # remove the "from" from the action string
            action = action.replace("from ", "")
        elif "move" in action:
            action_name = "move"
            # remove the "in or on" from the action string
            action = action.replace("in or on ", "")
        elif "clean" in action:
            action_name = "clean"
            # remove the "with" from the action string
            action = action.replace("with ", "")
        elif "heat" in action:
            action_name = "heat"
            # remove the "with" from the action string
            action = action.replace("with ", "")
        elif "cool" in action:
            action_name = "cool"
            # remove the "with" from the action string
            action = action.replace("with ", "")

        # remove the action name from the action string
        action = action.replace(action_name + " ", "")

        # get the action args
        action_args = action.split(" ")
        action_args.insert(0, action_name)

        return tuple(action_args)

    @classmethod
    def _get_starting_obj_env_mdata_list(cls, init_scene_obs: str) -> List[Dict]:
        """Given the Alfworld initial scene observation, generate a list of
        object metadata for each object in dictionary form.

        Args:
            init_scene_obs (str): the initial text description from the
                alfworld environment

        Returns:
            obj_mdata_list (List[Dict]): a list of dictionaries representing
                the object metadata of each object in the initial scene
                observation
        """
        # find the receptacles in the initial scene description
        receptacles = re.findall(r"a (\w+ \d+)", init_scene_obs)
        # make sure to sort the below if you want the environment facing object
        # ids to have corresponding index numbers to the llm facing ids
        receptacles = [recep.replace(" ", "-") for recep in receptacles]
        receptacles.sort()

        # get the object mdata list for the receptacles present in the initial
        # scene observation
        obj_mdata_list = list()
        for receptacle in receptacles:
            obj_mdata_list.append(cls._get_receptacle_attributes(receptacle))

        return obj_mdata_list

    @classmethod
    def _get_receptacle_attributes(cls, receptacle: str) -> Dict:
        """Alfworld does not provide the information of certain attributes.
        The LLM-DP Paper (the paper that we are basing this environment off
        of) manually adds attribute information for each receptacle. We will
        do the same thing with this function. This function is a direct
        copy-paste of the LLM-DP code (with a few naming changes.)

        Args:
            receptacle (str): the receptacle that we with to get attributes
                for

        Returns:
            obj_mdata (Dict): a dictionary representing the metadata of
                the object
        """
        obj_mdata = {}

        # predicate special types
        if "sink" in receptacle:
            obj_mdata["isSink"] = True
        elif "microwave" in receptacle:
            obj_mdata["isMicrowave"] = True
        elif "fridge" in receptacle:
            obj_mdata["isFridge"] = True

        # predicate openable
        if (
            "microwave" in receptacle
            or "fridge" in receptacle
            or "drawer" in receptacle
            or "cabinet" in receptacle
            or "safe" in receptacle
        ):
            obj_mdata["openable"] = True

        obj_mdata["objectId"] = receptacle
        obj_mdata["objectType"] = receptacle.split("-")[0]
        obj_mdata["isReceptacle"] = True

        return obj_mdata

    @classmethod
    def process_obs(cls, observation: str) -> dict:
        """
        Taken directly from the LLM-DP github code.
        """
        json_dict = {}
        # check if the receptacle is closed
        closed_receptacle = re.search(r"The (\w+ \d+) is closed", observation)
        if closed_receptacle:
            return json_dict
        # find the receptacle
        receptacle = re.search(
            r"(On|In) the (\w+ \d+)|You open the (\w+ \d+)", observation
        )
        if receptacle:
            # get the receptacle from the right group
            receptacle_key = (
                receptacle.group(2) if receptacle.group(2) else receptacle.group(3)
            )
            receptacle_key = receptacle_key.replace(" ", "-")
            json_dict[receptacle_key] = []
            # check if there's nothing in the receptacle
            no_items = re.search(r"you see nothing", observation)
            if no_items:
                return json_dict
            # find items in the receptacle
            items = re.findall(r"a (\w+ \d+)", observation)
            for item in items:
                json_dict[receptacle_key].append(item.replace(" ", "-"))
        return json_dict

    @classmethod
    def check_if_goal_state_reached(cls, global_task_metadata: str) -> bool:
        return cls.task_complete
