import asyncio
import json
import math
import os
import random
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.server import Event, MultiAgentEvent
from omegaconf import DictConfig
from scipy.spatial import distance

from src.tasks.ai2thor.tasks import AI2ThorTaskMetadata
from src.tasks.base import TaskMetadata
from src.utils.environment.base import Environment
from src.utils.misc.logging_utils import ensure_dir_exists, get_hydra_run_dir

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)

asyncio_lock = asyncio.Lock()


class AI2ThorEnv(Environment):

    is_controller_initialized: bool = False
    is_agent_states_initialized: bool = False
    c: Controller = Controller()
    cfg: DictConfig = DictConfig({})  # class config
    scene: str = None  # type: ignore
    agent_count: int = 1
    agent_states: Dict = dict()  # type: ignore
    # TODO: Maybe remove the below
    global_task_metadata: TaskMetadata = None  # type:ignore
    logging_dir: str = ""

    def __init__(self):
        if not self.is_controller_initialized:
            raise ValueError(
                "Must call AI2ThorEnv.initialize_ai2thor_controller(cfg, agent_count) before attempting to instantiate an object. See class docstring for more information."
            )

    @classmethod
    def initialize(cls, cfg: DictConfig, agent_count: int) -> None:
        """Initialize the ai2thor_controller, cfg, and agent count class
        variables. These variables are persistent across class instances.

        Args:
            cfg (DictConfig): the tool engine config
            agent_count (int): number of agents in the team (1 if single agent)

        Returns:
            None
        """
        cls.ai2thor_cfg = {
            "width": 600,
            "height": 600,
            "fieldOfView": 90,
            "agentMode": "default",
            "visibilityDistance": 1.5,
            "grid_size": 0.25,
            "snapToGrid": False,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "agentCount": agent_count,
        }
        if cfg.run_headless == True:
            cls.c.reset(platform=CloudRendering, **cls.ai2thor_cfg)
        else:
            cls.c.reset(**cls.ai2thor_cfg)
        cls.cfg = cfg
        cls.agent_count = agent_count

        cls.is_controller_initialized = True

    @classmethod
    def initialize_agents(cls, agents: List) -> None:
        """Initialize each agent."""
        # TODO: flesh this out..
        agent_ids = [a.agent_id for a in agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise Exception("Duplicate agent id found!")
        for agent in agents:
            cls.agent_states[agent.agent_id] = agent.state
            cls.agents[agent.agent_id] = agent
        cls.is_agent_states_initialized = True

    @classmethod
    def reset(cls, global_task_metadata: TaskMetadata):

        if not cls.is_agent_states_initialized:
            raise ValueError(
                "Must call AI2ThorEnv.initialize_agents(agents) before attempting to instantiate an object. See class docstring for more information."
            )

        if not cls.is_controller_initialized:
            raise ValueError(
                "Error: Must call AI2ThorEnv.initialize_ai2thor_controller(cfg, agent_count) before attempting to instantiate an object. See class docstring for more information."
            )

        cls.global_task_metadata = global_task_metadata
        cls.scene = global_task_metadata.scene
        cls.c.reset(scene=cls.scene, **cls.ai2thor_cfg)

        # set overhead camera
        event = cls.c.step(action="GetMapViewCameraProperties")
        cls.c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

        # reset objects to start with objects that are not hiding inside of
        # other objects (i.e. they have a parent receptacle that is openable)
        cls.obj_db.reset()
        objs_not_in_receptacle = asyncio.run(cls.get_objs_not_in_openable_receptacle())
        [cls.obj_db.add_obj(x) for x in objs_not_in_receptacle]  # so pythonic :)

        # cause the agent to look down so that it can see the objects that it is interacting with
        for agent_id in range(AI2ThorEnv.agent_count):
            cls.c.step(action="LookDown", degrees=35, agentId=agent_id)

        # reset agent states and set current location to random object
        for _, a_state in cls.agent_states.items():
            a_state.reset()
            a_state.cur_loc = random.choice(list(cls.obj_db.llm_ids))

        # update the default valid receptacles for each discovered object
        cls.add_default_valid_receptacles()

        cls.env_steps: list = list()

        cls.at_location = None

    @classmethod
    def set_logging_dir(cls, logging_dir: str) -> None:
        cls.logging_dir = logging_dir

    @classmethod
    async def populate_obj_db_w_ground_truth(cls) -> None:
        """There may be a time where we want to have an object database that
        contains every single object in the AI2Thor simulator environment
        even if some objects have not been discovered (for example, for checking
        the overall goal state.) This function will populate the object database
        with any remaining undiscovered objects to provide an object database
        representing the ground truth for that environment."""
        obj_mdata_list = await cls.get_obj_mdata_list()
        for obj_mdata in obj_mdata_list:
            if obj_mdata["objectId"] not in cls.obj_db.env_ids:
                cls.obj_db.add_obj(obj_mdata)

    @classmethod
    async def step(
        cls, agent_id: int, save_img: bool = False, iterate_steps: bool = False, **kwargs
    ) -> Event | MultiAgentEvent:
        """Safely calls an AI2Thor controller while locking the async
        calling to avoid errors from multiple threads calling the same
        AI2Thor controller."""
        async with asyncio_lock:
            
            if iterate_steps:
                cls.env_steps.append(kwargs)

            if "objectId" in kwargs:
                # there is a weird error that does not allow you to interact with
                # a sink. instead, you must interact with the basin. this is a hack
                # for the bug
                if "Sink" in kwargs["objectId"]:
                    kwargs["objectId"] = cls.obj_db.ensure_env_id("SinkBasin_1")

            # check if there are slilces of something in the sink basin
            slices_objs_in_sink_basin = None
            for obj_id, obj_mdata in cls.obj_db.obj_metadatas.items():
                if "SinkBasin" in obj_id:
                    objs_in_sink_basin = obj_mdata["receptacleObjectIds"]
                    if isinstance(objs_in_sink_basin, list) and len(objs_in_sink_basin) > 0:
                        slices_objs_in_sink_basin = [x for x in objs_in_sink_basin if "slice" in x.lower()]

            # if picking up an object, or moving with a held object, you must
            # check to see if the object is filled with a liquid. there is a bug
            # in ai2thor that removes any liquid from a held object when teleporting
            # or picking that object up
            fill_liquid = -1
            held_obj = None
            for obj_id, obj_mdata in cls.obj_db.obj_metadatas.items():
                if obj_mdata["isPickedUp"]:
                    held_obj = obj_id
                    fill_liquid = obj_mdata["fillLiquid"]

            if "action" in kwargs and kwargs["action"] == "PickupObject":
                obj_mdata = cls.obj_db.get_obj_metadata(kwargs["objectId"])
                assert fill_liquid == -1
                fill_liquid = obj_mdata["fillLiquid"]

            event = cls.c.step(
                **kwargs, agentId=agent_id
            )  # Run the coroutine and wait for the result

            # if the agent picks something up or teleports, removing the filled
            # liquid from the container, re-fill the container with the correct
            # liquid
            if (
                fill_liquid is not None
                and fill_liquid != -1
                and event.metadata["lastActionSuccess"]
            ):
                if held_obj is None:
                    held_obj = cls.obj_db.ensure_env_id(kwargs["objectId"])
                _ = cls.c.step(
                    action="FillObjectWithLiquid",
                    objectId=held_obj,
                    fillLiquid=fill_liquid,
                    forceAction=True,
                )

            # a 'do nothing' step to update the visualization
            _ = cls.c.step(action="MoveAhead", moveMagnitude=0, agentId=agent_id)

            # add new objects that are visible
            obj_mdata_list = event.metadata["objects"]
            for obj_mdata in obj_mdata_list:
                if (
                    obj_mdata["visible"]
                    and obj_mdata["objectId"] not in cls.obj_db.env_ids
                ):
                    cls.obj_db.add_obj(obj_mdata)

            # update the object metadata of all objects in the scene
            cls.obj_db.update_obj_metadata_list(obj_mdata_list)

            # this will update the list of valid receptacles
            cls.add_default_valid_receptacles()

            # save the overhead images
            if save_img:
                cls._save_top_view_img()
                cls._save_front_view_img()

            return event

    @classmethod
    def add_default_valid_receptacles(cls):
        """AI2Thor gives a list of default receptacles for each object (unsliced
        bread cannot be placed in a toaster but can be placed in a cabinet). This
        function goes through and updates all of the object metadatas to include
        valid receptacles."""
        obj_type_mdata_json_path = os.path.join(
            CUR_DIR, "ai2thor_metadata", "object_type_metadata.json"
        )
        with open(obj_type_mdata_json_path, "r") as file:
            obj_type_mdata = json.load(file)
        for obj_id, obj_mdata in cls.obj_db.obj_metadatas.items():
            obj_type = obj_mdata["objectType"]
            default_compatible_receptacles = set(
                obj_type_mdata[obj_type]["Default_Compatible_Receptacles"]
            )
            cls.obj_db.obj_metadatas[obj_id]["valid_receptacles"] = list()
            for _obj_id, _obj_mdata in cls.obj_db.obj_metadatas.items():
                if _obj_mdata["objectType"] in default_compatible_receptacles:
                    cls.obj_db.obj_metadatas[obj_id]["valid_receptacles"].append(
                        _obj_id
                    )

    @classmethod
    async def get_obj_mdata_from_obj_id(cls, ai2thor_obj_id: str) -> Dict:
        """Returns the object metadata for an object specified by ai2thor
        object id. If object id not found int the object metadata list, will
        return None.

        Args:
            obj_id (str): the ai2thor object id of the object that you wish to
                return the metadata for

        Returns:
            object_metadata (Dict): Returns dictionary of object metadata
                otherwise
        """
        obj_mdata_list = await cls.get_obj_mdata_list()
        for obj_mdata in obj_mdata_list:
            if ai2thor_obj_id == obj_mdata["objectId"]:
                return obj_mdata
        raise ValueError(
            f"Error: {ai2thor_obj_id} not found in AI2Thor object metadata."
        )

    @classmethod
    async def get_obj_pos(cls, obj_id: str) -> Dict:
        """Returns the position of an object in the ai2thor sim.

        Args:
            obj_id (str): ai2thor object id

        Returns:
            pos (Dict): Returns the 3d position of the object as a dict with
                keys x, y, and z
        """
        obj_mdata = await cls.get_obj_mdata_from_obj_id(obj_id)
        return obj_mdata["axisAlignedBoundingBox"]["center"]

    @classmethod
    def sort_nodes_by_distance(cls, nodes: np.ndarray, target_pos: np.ndarray):
        """Given a list of nodes with positions and a target position,
        sort the nodes in terms of distance from the target position.

        Args:
            nodes (np.ndarray): a numpy array of shape (N, 3) providing the x,
                y, and z positions of the nodes
            pos (np.ndarray): a numpy array of shape (1, 3) providing the x, y,
            and z positions of the nodes

        Return:
            nodes_closest_to_farthest (np.ndarray): array of shape (N, 3) of
                the list of nodes sorted in ascending order in terms of
                distance from the target position
        """
        distances = distance.cdist(target_pos, nodes)[0]
        sort_indices = np.argsort(np.array(distances))
        nodes_closest_to_farthest = nodes[sort_indices]
        return nodes_closest_to_farthest

    @classmethod
    async def move_to_object(
        cls, agent_id: int, obj_id: str
    ) -> Event | MultiAgentEvent:
        """Moves to an object in the AI2Thor environment and positions the
        object in the middle of the camera frame.

        Args:
            agent_id (int): the id of the agent
            obj_id (str): the ai2thor object id that you wish to move to

        Returns:
            event (Event | MultiAgnetEvent): the return ai2thor event from
                the action
        """
        # get a list of valid poses in which the object is visible
        event = await cls.step(
            agent_id,
            action="GetInteractablePoses",
            objectId=obj_id,
            horizons=np.linspace(-30, 60, 5),
        )
        poses = event.metadata["actionReturn"]
        if len(poses) == 0:
            raise ValueError("No interactable poses found.")
        # get the position of the object in the environment
        obj_pos = await cls.get_obj_pos(obj_id)
        # get the pose closest to the target object
        poses_np = np.array([[i["x"], i["y"], i["z"]] for i in poses])
        obj_pos_np = np.array([obj_pos["x"], obj_pos["y"], obj_pos["z"]])
        obj_pos_np = np.expand_dims(obj_pos_np, axis=0)
        poses_closest_to_farthest = cls.sort_nodes_by_distance(poses_np, obj_pos_np)

        # TODO: Determine if the poses includes poses that are currently
        # occupied by other robots. If so, will need to account for this.

        # loop through closest poses until successfully teleports
        is_successful_teleport = False
        for pose in poses_closest_to_farthest:
            x, y, z = pose
            event = await cls.step(
                agent_id, action="Teleport", position=dict(x=x, y=y, z=z), horizon=35
            )
            if event.metadata["lastActionSuccess"]:  # type:ignore
                is_successful_teleport = True
                break
        if not is_successful_teleport:
            raise ValueError("Teleport not successful.")
        if cls.c.last_event is None:
            raise ValueError("Controller must have last event.")
        # rotate agent to look at object
        # currently this just turns right or left, does not tilt the camera up or
        # down as this has proved unreliable as the y position of ai2thor objects
        # are inconsistent (sometimes -y is up, sometimes -y is down)
        metadata = cls.c.last_event.events[agent_id].metadata
        agent_pos = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
        }
        agent_obj_vector = np.array(
            [
                obj_pos_np[0, 0] - agent_pos["x"],
                obj_pos_np[0, 2] - agent_pos["z"],
            ]
        )
        y_axis = [0, 1]
        unit_y = y_axis / np.linalg.norm(y_axis)  # type:ignore
        unit_vector = agent_obj_vector / np.linalg.norm(agent_obj_vector)
        angle = math.atan2(
            np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y)
        )
        angle = 360 * angle / (2 * np.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - agent_pos["rotation"]
        if rot_angle > 0:
            event = await cls.step(
                agent_id, action="RotateRight", degrees=abs(rot_angle)
            )
        else:
            event = await cls.step(
                agent_id, action="RotateLeft", degrees=abs(rot_angle)
            )
        # modify horizon angle if needed
        # get the current agent position
        metadata = cls.c.last_event.events[agent_id].metadata
        agent_pos = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
        }
        # determine all horizon angles that can be seen at the agents current
        # position and rotation
        event = await cls.step(
            agent_id,
            action="GetInteractablePoses",
            objectId=obj_id,
            positions=[dict(x=agent_pos["x"], y=0.9, z=agent_pos["z"])],
            rotations=[agent_pos["rotation"]],
            horizons=np.linspace(-30, 60, 80),
        )
        new_poses = event.metadata["actionReturn"]
        valid_horizon_angles = np.array(
            [x["horizon"] for x in new_poses if x["standing"]]
        )
        # take the average of the valid horizon angles
        target_horizon = np.median(valid_horizon_angles)

        target_pose = {
            "x": agent_pos["x"],
            "y": agent_pos["y"],
            "z": agent_pos["z"],
            "rotation": agent_pos["rotation"],
            "standing": True,
            "horizon": target_horizon,
        }

        event = await cls.step(
            agent_id, 
            action="Teleport", 
            save_img=True, 
            iterate_steps=True, 
            **target_pose
        )

        return event

    @classmethod
    async def get_obj_mdata_list(cls) -> List[Dict]:
        """Returns a list of the metadata of each object in the ai2thor
        simulation.

        Args:
            None

        Returns:
            obj_mdata_list (List[Dict]): a list of the object metadata
                for each object in the ai2thor simulator
        """
        if cls.c.last_event is None:
            # ensure the controller has a last event from which to get information
            await cls.step(0, action="MoveAhead", moveMagnitude=0)
        if cls.c.last_event is None:
            raise ValueError("Controller must have last event.")
        obj_mdata_list = cls.c.last_event.metadata["objects"]
        return obj_mdata_list

    @classmethod
    async def get_objs_not_in_openable_receptacle(cls) -> List[Dict]:
        """Returns a list of object metadatas for objects that are
        currenly not inside a receptacle that can be opened. Other
        receptacles, like countertops and shelves, cannot be opened,
        thus the objects on top of these receptacles are not hidden.

        Args:
            None

        Returns:
            objs_not_in_receptacles (List[Dict]): a list of object metadatas
                corresponding to objects not inside a receptacle
        """
        objs_mdata_list = await cls.get_obj_mdata_list()
        objs_not_in_openable_receptacle = []
        for mdata in objs_mdata_list:
            if (
                mdata["parentReceptacles"] is None
                or len(mdata["parentReceptacles"]) == 0
            ):
                objs_not_in_openable_receptacle.append(mdata)
                continue
            # has parent receptacles. need to check if any of them are openable
            # if not, then the object is not hidden, so we add it the the list
            parent_receptacle_openable = False
            for receptacle in mdata["parentReceptacles"]:
                receptacle_obj_mdata = [
                    x for x in objs_mdata_list if x["objectId"] == receptacle
                ][0]
                if receptacle_obj_mdata["openable"]:
                    parent_receptacle_openable = True
                    break
            if not parent_receptacle_openable:
                objs_not_in_openable_receptacle.append(mdata)
        return objs_not_in_openable_receptacle

    @classmethod
    def _save_top_view_img(cls) -> None:
        """Saves the top view image from the ai2thor simulator."""
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        top_view_rgb = cv2.cvtColor(
            cls.c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB
        )
        if (
            cls.logging_dir == ""
            or cls.global_task_metadata.task_name.lower() not in cls.logging_dir.lower()
        ):
            raise Exception(
                "Logging directory either not initialized or does not match global task metadata."
            )
        img_save_dir = os.path.join(cls.logging_dir, "top_view_imgs")
        ensure_dir_exists(img_save_dir)
        f_name = os.path.join(img_save_dir, "img_" + formatted_time + ".png")
        cv2.imwrite(f_name, top_view_rgb)

    @classmethod
    def _save_front_view_img(cls) -> None:
        """Saves the front view image from the ai2thor simulator."""
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        if isinstance(cls.c.last_event, MultiAgentEvent):
            front_view_rgb = cls.c.last_event.cv2img
        else:
            front_view_rgb = cls.c.last_event.cv2img
        if (
            cls.logging_dir == ""
            or cls.global_task_metadata.task_name.lower() not in cls.logging_dir.lower()
        ):
            raise Exception(
                "Logging directory either not initialized or does not match global task metadata."
            )
        img_save_dir = os.path.join(cls.logging_dir, "front_view_imgs")
        ensure_dir_exists(img_save_dir)
        f_name = os.path.join(img_save_dir, "img_" + formatted_time + ".png")
        cv2.imwrite(f_name, front_view_rgb)

    @classmethod
    def check_if_goal_state_reached(cls, task_metadata: AI2ThorTaskMetadata) -> bool | str:
        return task_metadata.verification_func(list(cls.obj_db.obj_metadatas.values()))
    
    @classmethod
    def check_if_holding_object(cls) -> bool:
        for obj_id, obj_mdata in cls.obj_db.obj_metadatas.items():
            if obj_mdata["isPickedUp"]:
                return True
        return False
