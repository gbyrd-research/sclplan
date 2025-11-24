"""The verification functions are kept in this python file
and are intended to be imported into the tasks.py file."""

import glob
import json
import os
from typing import Dict, List

from src.utils.misc.formatting import format_pddl_goal_state
from src.utils.state.goal_state_functions import (
    ensure_obj_id_in_obj_type,
    ensure_obj_type_has_att,
    ensure_obj_type_in_obj_type,
)
from src.utils.state.object_database import ObjectDatabase

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)
SCENE_OBJ_METADATA_DIR = os.path.join(
    CUR_DIR, "..", "..", "utils", "environment", "ai2thor", "ai2thor_metadata", "scene_object_metadata"
)


def get_obj_mdata_list(floorplan: str) -> List[Dict]:
    # search for floor plan metadata
    obj_mdata_json_files = glob.glob(os.path.join(SCENE_OBJ_METADATA_DIR, "*.json"))
    matching_obj_mdata_json_files = [
        x for x in obj_mdata_json_files if floorplan.lower() + "_" in x.lower()
    ]
    if len(matching_obj_mdata_json_files) == 0:
        raise ValueError(f"No object metadata list for floorplan: {floorplan} found.")
    if len(matching_obj_mdata_json_files) > 1:
        raise Exception(
            f"There are duplicate object metadata lists for floorplan: {floorplan}."
        )
    # load the object metadata
    with open(matching_obj_mdata_json_files[0], "r") as file:
        obj_mdata_list = json.load(file)

    return obj_mdata_list


def get_obj_db(floorplan: str) -> ObjectDatabase:
    """Given a list of object metadata, return an ObjectDatabase
    corresponding to the list of data.

    Args:
        floorplan (str): a string representing the name of the
            desired floorplan to get a populated object database
            of

    Returns:
        obj_db (ObjectDatabase): a populated object database with
            all objects present in the floorplan
    """
    obj_mdata_list = get_obj_mdata_list(floorplan)
    obj_db = ObjectDatabase()
    for obj_mdata in obj_mdata_list:
        obj_db.add_obj(obj_mdata)

    return obj_db


# each function can create a goal state for any floorplan, although some
# tasks do not work for different floorplans. for example, "Cook an egg"
# will not work for a floor plan that does not contain an egg object
def task_0000_goal_state(floorplan: str):
    obj_db = get_obj_db(floorplan)
    valid_states = ensure_obj_type_has_att(obj_db, "egg", "isCooked", True)
    return format_pddl_goal_state(valid_states)


def task_0001_goal_state(floorplan: str):
    obj_db = get_obj_db(floorplan)
    valid_states = list()
    valid_states.extend(ensure_obj_type_in_obj_type(obj_db, "kettle", "stoveburner"))
    valid_states.extend(ensure_obj_type_in_obj_type(obj_db, "apple", "fridge"))
    valid_states.extend(ensure_obj_type_has_att(obj_db, "bread", "isSliced", True))
    return format_pddl_goal_state(valid_states)


def task_0002_goal_state(floorplan: str):
    obj_db = get_obj_db(floorplan)
    valid_states = list()
    valid_states.extend(ensure_obj_type_in_obj_type(obj_db, "kettle", "stoveburner"))
    valid_states.extend(ensure_obj_type_has_att(obj_db, "bread", "isSliced", "True"))
    valid_states.extend(ensure_obj_type_in_obj_type(obj_db, "egg", "bowl"))
    return format_pddl_goal_state(valid_states)


def task_0003_goal_state(floorplan: str):
    obj_db = get_obj_db(floorplan)
    valid_states = list()
    valid_states.extend(ensure_obj_type_in_obj_type(obj_db, "egg", "bowl"))
    return format_pddl_goal_state(valid_states)
