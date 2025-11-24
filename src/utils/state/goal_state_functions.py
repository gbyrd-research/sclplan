import warnings
from typing import Callable, Dict, List

from src.utils.state.object_database import ObjectDatabase

valid_pddl_predicates = {
    "visible",
    "isInteractable",
    "toggleable",
    "isToggled",
    "breakable",
    "isBroken",
    "canFillWithLiquid",
    "isFilledWithLiquid",
    "dirtyable",
    "isDirty",
    "canBeUsedUp",
    "isUsedUp",
    "cookable",
    "isCooked",
    "isHeatSource",
    "isColdSource",
    "sliceable",
    "isSliced",
    "isKnife",
    "openable",
    "isOpen",
    "pickupable",
    "isPickedUp",
    "moveable",
    "isHoldingObject",
    "objectInReceptacle",
    "objectHeld",
    "agentInLocation",
    "atLocation",
}


def or_pddl(func: Callable):

    def wrapper(*args, **kwargs) -> List[str]:
        """Takes in a list of valid states and adds an entry at
        the beginning and end of the list to wrap the list in a
        valid PDDL or block."""

        valid_states = func(*args, **kwargs)

        new_valid_states = list()
        # add tab to valid states
        new_valid_states.extend(["\t" + x for x in valid_states])
        new_valid_states.insert(0, f"(or \n")
        new_valid_states.append(")\n")
        return new_valid_states

    return wrapper


def and_pddl(func: Callable):

    def wrapper(*args, **kwargs) -> List[str]:
        """Takes in a list of valid states and adds an entry at
        the beginning and end of the list to wrap the list in a
        valid PDDL and block."""

        valid_states = func(*args, **kwargs)

        new_valid_states = list()
        # add tab to valid states
        new_valid_states.extend(["\t" + x for x in valid_states])
        new_valid_states.insert(0, f"(and \n")
        new_valid_states.append(")\n")
        return new_valid_states

    return wrapper


@or_pddl
def ensure_obj_type_in_obj_type(
    obj_db: ObjectDatabase, obj_type: str, receptacle_type: str
) -> List[str]:
    """Returns a goal state that will be true if the object with specified object
    id is inside a receptacle with specified receptacle type.
    """

    obj_metadata_list = obj_db.get_obj_metadata_list()

    # sometimes an object id is provided in place of a receptacle type
    # in the event that this occurs, we will extract the object type from the
    # object id
    receptacle_type = receptacle_type.replace(" ", "")
    if "_" in receptacle_type:
        receptacle_type = receptacle_type.split("_")[0]

    valid_objects = [
        obj_db.env_to_llm(x["objectId"])
        for x in obj_metadata_list
        if obj_type.lower() in x["objectId"].lower()
    ]

    valid_receptacles = [
        obj_db.env_to_llm(x["objectId"])
        for x in obj_metadata_list
        if receptacle_type.lower() in x["objectId"].lower()
    ]

    valid_states = list()
    for o in valid_objects:
        for r in valid_receptacles:
            valid_states.append(f"(objectInReceptacle {o} {r})\n")
    return valid_states


@or_pddl
def ensure_obj_id_in_obj_type(
    obj_db: ObjectDatabase, llm_obj_id: str, receptacle_type: str
) -> List[str]:
    """Returns a goal state that will be true if the object with specified object
    id is inside a receptacle with specified receptacle type.
    """

    obj_metadata_list = obj_db.get_obj_metadata_list()

    # sometimes an object id is provided in place of a receptacle type
    # in the event that this occurs, we will extract the object type from the
    # object id
    receptacle_type = receptacle_type.replace(" ", "")
    if "_" in receptacle_type:
        receptacle_type = receptacle_type.split("_")[0]

    valid_objects = [llm_obj_id]

    valid_receptacles = [
        obj_db.env_to_llm(x["objectId"])
        for x in obj_metadata_list
        if receptacle_type.lower() in x["objectId"].lower()
    ]

    valid_states = list()
    for o in valid_objects:
        for r in valid_receptacles:
            valid_states.append(f"(objectInReceptacle {o} {r})\n")
    return valid_states


def ensure_obj_type_has_att(
    obj_db: ObjectDatabase, obj_type: str, att: str, att_val: bool
) -> List[str]:
    """Returns a pddl goal state that will be true if any object with
    specified object type has the specified attribute with specified
    value."""

    obj_metadata_list = obj_db.get_obj_metadata_list()

    obj_type = obj_type.replace(" ", "")

    if att not in valid_pddl_predicates:
        raise KeyError("Att must match a valid pddl predicate.")

    valid_objects = [
        obj_db.env_to_llm(x["objectId"])
        for x in obj_metadata_list
        if obj_type.lower() in x["objectId"].lower()
    ]

    valid_states = list()
    for o in valid_objects:
        if att_val:
            valid_states.append(f"({att} {o})\n")
        else:
            valid_states.append(f"(not ({att} {o}))\n")
    return valid_states


def ensure_obj_id_has_att(
    obj_db: ObjectDatabase, llm_obj_id: str, att: str, att_val: bool
) -> List[str]:
    """Returns a pddl goal state that will be true if any object with
    specified object id has the specified attribute with specified value."""
    if att not in valid_pddl_predicates:
        warnings.warn(f"{att} not valid. Must be one of {valid_pddl_predicates}.")

    obj_metadata_list = obj_db.get_obj_metadata_list()

    # we just want a warning here, not an exception, because we want the LLM
    # to be able to propose goal states that may have objects that have not yet
    # been discovered in the environment
    if llm_obj_id not in [
        obj_db.env_to_llm(x["objectId"]) for x in obj_metadata_list
    ]:
        warnings.warn(f"No object with id {llm_obj_id} in the AI2Thor scene.")
    if llm_obj_id not in [x for x in obj_db.llm_ids]:
        warnings.warn(
            f"No object with id {llm_obj_id} discovered yet in the AI2Thor scene."
        )

    valid_states = list()
    if att_val:
        valid_states.append(f"({att} {llm_obj_id})")
    else:
        valid_states.append(f"(not ({att} {llm_obj_id}))")
    return valid_states


def get_goal_state_helper_descs() -> Dict:
    # some LLM agents in this repository use these functions to more easily specify
    # goal states at a high level. As such, we need to provide a description of
    # each goal state definition helper functions in natural language
    goal_state_helper_descriptions = {
        "object_id_in_object_type": {
            "desc": "object_id_in_object_type(object_id: str, receptacle_type: str) - Defines a goal state in which the specified object id is inside or on top of the specified receptacle.",
            "arg_types": (str, str),
            "func": ensure_obj_id_in_obj_type,
        },
        "object_id_has_attribute": {
            "desc": "object_id_has_attribute(object_id: str, attribute: str, attribute_value: bool) - Defines a goal state in which the specified object id has the specified value for a specified attribute.",
            "arg_types": (str, str, bool),
            "func": ensure_obj_id_has_att,
        },
    }
    return goal_state_helper_descriptions
