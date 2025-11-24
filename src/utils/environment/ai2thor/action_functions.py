"""
Define all action call functions for the AI2Thor environment.
"""

from typing import Tuple

from src.utils.environment.ai2thor.env import AI2ThorEnv


async def move_to_object(object_id: str, agent_id: int) -> Tuple[int, str]:
    """Move an agent to an object in the ai2thor simulator.

    Args:
        object_id (str): the llm facing object id that you wish to move to

    Returns:
        flag (int): the flag specifying the result of the tool call
        obj (str): the string representing the observation of the
            tool call
    """
    # TODO: Make it so that the agent does not simply teleport to the
    # object and instead will incrementally move towards the object.
    # this will make for better visuals.
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(object_id)

    """
    PRECONDITIONS
        Although there are preconditions in the pddl representation of
        this tool, these preconditions are not present in the tool API
        calls because I have not thought of a good way to verify them.
    """

    event = await AI2ThorEnv.step(
        agent_id,
        action="GetInteractablePoses",
        objectId=ai2thor_obj_id
    )
    poses = event.metadata["actionReturn"]
    if len(poses) == 0:
        return 3, f"Could not move to {object_id}."
    event = await AI2ThorEnv.move_to_object(agent_id, ai2thor_obj_id)
    if not event.metadata["lastActionSuccess"]:
        if "Cannot teleport due to hand object collision" in event.metadata["errorMessage"]:
            raise ValueError("Cannot move to location due to collision with held object. Must place held object down before moving.")
        raise ValueError("Last action should have completed successfully.")
    AI2ThorEnv.agent_states[agent_id].cur_loc = AI2ThorEnv.obj_db.ensure_llm_id(object_id)
    return 0, f"Moved to {object_id}."


async def open_object(object_id: str, agent_id: int) -> Tuple[int, str]:
    """Open an object. If there are objects inside the opened object,
    add the object metadata of these objects to the list of objects
    and return an observation detailing which objects were found.
    """
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(object_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)

    state = AI2ThorEnv.agent_states[agent_id]

    """
    PRECONDITIONS
        the agent must be at the target object
        the target object must be openable
        the target object must not already be open
        the agent must not already by holding an object
    """
    # sometimes a held object can occlude the target object
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(object_id):
        return 3, f"{object_id} is not visible."
    if not targ_obj_mdata["openable"]:
        return 3, f"{object_id} is not openable."
    if targ_obj_mdata["isOpen"]:
        return 3, f"{object_id} is already open."
    if state.held_obj is not None:
        return 3, f"Cannot open {object_id} while holding {state.held_obj}."

    # all preconditions satisfied
    event = await AI2ThorEnv.step(
        agent_id, 
        action="OpenObject", 
        objectId=ai2thor_obj_id, 
        save_img=True, 
        iterate_steps=True
    )
    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not open {object_id}."
    # if objects were found inside the opened object, 1) add them to the list,
    # 2) return an observation notifying the agent of the found objects
    obj_metadata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)
    found_ids = obj_metadata["receptacleObjectIds"]
    found_ids = [] if found_ids is None else found_ids
    for id in found_ids:
        temp_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(id)
        AI2ThorEnv.obj_db.add_obj(temp_mdata)
    llm_found_ids = [AI2ThorEnv.obj_db.get_llm_from_env(x) for x in found_ids]
    if len(llm_found_ids) > 0:
        found_objects = ", ".join(llm_found_ids)
        return (
            0,
            f"Opened {object_id}. Found the following objects: {found_objects}",
        )
    return 0, f"Opened {object_id}."

async def close_object(object_id: str, agent_id: int) -> Tuple[int, str]:
    """Open an object. If there are objects inside the opened object,
    add the object metadata of these objects to the list of objects
    and return an observation detailing which objects were found.
    """
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(object_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)

    state = AI2ThorEnv.agent_states[agent_id]

    """
    PRECONDITIONS
        the agent must be at the target object
        the target object must be openable
        the target object must already be open
        the agent must not already by holding an object
    """
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(object_id):
        return 3, f"{object_id} is not visible."
    if not targ_obj_mdata["openable"]:
        return 3, f"{object_id} is not openable."
    if not targ_obj_mdata["isOpen"]:
        return 3, f"{object_id} is already closed."
    if state.held_obj is not None:
        return 3, f"Cannot open {object_id} while holding {state.held_obj}."

    # all preconditions satisfied
    event = await AI2ThorEnv.step(
        agent_id, 
        action="CloseObject", 
        objectId=ai2thor_obj_id, 
        save_img=True, 
        iterate_steps=True
    )
    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not close {object_id}."
    return 0, f"Closed {object_id}."


async def pickup_object(object_id: str, agent_id: int) -> Tuple[int, str]:
    """Pick up an object. If an object is already picked up,
    return failure."""
    state = AI2ThorEnv.agent_states[agent_id]
    if state.held_obj is not None:
        return 3, f"Already holding an object."
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(object_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)

    """
    PRECONDITIONS
        the agent must be at the target object
        the target object must be able to be picked up
        the agent must not already be holding an object
        the target object must not already by picked up
    """
    # sometimes a held object can occlude the target object
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(object_id):
        return 3, f"{object_id} is not visible."
    if not targ_obj_mdata["pickupable"]:
        return 3, f"{object_id} is not pickupable."
    if state.held_obj:
        return (
            3,
            f"Agent already holding {state.held_obj}. Cannot pick up {object_id}.",
        )
    if targ_obj_mdata["isPickedUp"]:
        return 3, f"{object_id} is already picked up by another agent!"

    # all preconditions satisfied. attempt to pick up object
    event = await AI2ThorEnv.step(
        agent_id, 
        action="PickupObject", 
        objectId=ai2thor_obj_id, 
        save_img=True, 
        iterate_steps=True
    )
    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not pick up {object_id}."
    # if it gets here, the agent has successfully picked up the object
    state.held_obj = object_id
    return 0, f"Picked up {object_id}."


async def place_object(held_object_id: str, receptacle_id: str, agent_id: int) -> Tuple[int, str]:
    """Places the held object in or on an object with specified
    object id. If an object is not already held, return failure."""
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(receptacle_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)
    state = AI2ThorEnv.agent_states[agent_id]

    """
    PRECONDITIONS
        the agent must be holding an object to place
        the target receptacle must actually be a receptacle
        the agent must be at the target receptacle
        (or (the target receptacle must not be openable)
            (the target receptacle is open)
        )
    """
    if "Sink" in receptacle_id:
        receptacle_id = AI2ThorEnv.obj_db.ensure_llm_id("SinkBasin_1")

    if state.held_obj != held_object_id:
        return 3, f"Not holding {held_object_id} therefore cannot place."
    if not targ_obj_mdata["receptacle"]:
        return 3, f"{receptacle_id} not a valid receptacle."
    # sometimes a held object can occlude the target object
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(receptacle_id):
        return 3, f"{receptacle_id} is not visible."
    if targ_obj_mdata["openable"] and not targ_obj_mdata["isOpen"]:
        return (
            3,
            f"Cannot place {held_object_id} inside {receptacle_id} because it is not open.",
        )

    # preconditions satisfied. attempt to place the held object in or
    # on the target object
    event = await AI2ThorEnv.step(
        agent_id,
        action="PutObject",
        objectId=ai2thor_obj_id,
        forceAction=True,
        save_img=True,
        iterate_steps=True
    )
    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not place {state.held_obj} in or on {receptacle_id}."
    # if it gets here, the agent has successfully placed its held object
    previously_held_obj = state.held_obj
    state.held_obj = None
    return 0, f"Placed {previously_held_obj} in or on {receptacle_id}."


async def toggle_object_on(object_id: str, agent_id: int) -> Tuple[int, str]:
    """Attempts to toggle an object on. If an object is already
    held, or if the object is not toggleable, or if the object
    is already toggled on, return failure."""

    state = AI2ThorEnv.agent_states[agent_id]
    if state.held_obj is not None:
        return (
            3,
            f"Cannot toggle on {object_id} while holding {state.held_obj}.",
        )
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(object_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)

    """
    PRECONDITIONS
        the target object must be togglable
        the target agent must be at the target object
        the target object must be toggled off
        the target agent must not be holding something
    """
    if not targ_obj_mdata["toggleable"]:
        return 3, f"{object_id} is not toggleable."
    # sometimes a held object can occlude the target object
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(object_id):
        return 3, f"{object_id} is not visible."
    if targ_obj_mdata["isToggled"]:
        return 3, f"{object_id} already toggled on."
    if state.held_obj:
        return 3, f"Holding {state.held_obj}. Cannot toggle on {object_id}."

    # all checks passed. attempt to toggle on
    event = await AI2ThorEnv.step(
        agent_id, 
        action="ToggleObjectOn", 
        objectId=ai2thor_obj_id, 
        save_img=True, 
        iterate_steps=True
    )
    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not toggle on {object_id}."
    return 0, f"Toggled on {object_id}."


async def toggle_object_off(object_id: str, agent_id: int) -> Tuple[int, str]:
    """Attempts to toggle an object off. If an object is already
    held, or if the object is not toggleable, or if the object
    is already toggled off, return failure."""
    state = AI2ThorEnv.agent_states[agent_id]
    if state.held_obj is not None:
        return (
            3,
            f"Cannot toggle off {object_id} while holding {state.held_obj}.",
        )
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(object_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)

    """
    PRECONDITIONS
        the target object must be togglable
        the target agent must be at the target object
        the target object must be toggled on
        the target agent must not be holding something
    """
    if not targ_obj_mdata["toggleable"]:
        return 3, f"{object_id} is not toggleable."
    # check if target is visible
    # sometimes a held object can occlude the target object
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(object_id):
        return 3, f"{object_id} is not visible."
    if not targ_obj_mdata["isToggled"]:
        return 3, f"{object_id} already toggled off."
    if state.held_obj:
        return 3, f"Holding {state.held_obj}. Cannot toggle on {object_id}."

    # all checks passed. attempt to toggle on
    event = await AI2ThorEnv.step(
        agent_id, 
        action="ToggleObjectOff", 
        objectId=ai2thor_obj_id,
        save_img=True, 
        iterate_steps=True
    )
    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not toggle off {object_id}."
    return 0, f"Toggled off {object_id}."


async def slice_object(slicing_object_id: str, target_object_id: str, agent_id: int) -> Tuple[int, str]:
    """Attempts to slice an object. If the object is not holding a
    knife, or the target object is not sliceable, return failure.
    """
    state = AI2ThorEnv.agent_states[agent_id]
    if state.held_obj != slicing_object_id:
        return 3, f"Cannot slice {target_object_id}. Not holding {slicing_object_id}."
    ai2thor_obj_id = AI2ThorEnv.obj_db.get_env_from_llm(target_object_id)
    targ_obj_mdata = await AI2ThorEnv.get_obj_mdata_from_obj_id(ai2thor_obj_id)

    """
    PRECONDITIONS
        the target agent must be holding an knife
        the target agent must be at the location of the object to be sliced
        the target object to be sliced must be sliceable
        the target object to be sliced must not already be sliced
    """
    if not state.held_obj or "knife" not in state.held_obj.lower():
        return 3, f"Not holding a knife. Cannot slice {target_object_id}."
    # sometimes a held object can occlude the target object
    if AI2ThorEnv.agent_states[agent_id].cur_loc != AI2ThorEnv.obj_db.ensure_llm_id(target_object_id):
        return 3, f"{target_object_id} is not visible."
    if not targ_obj_mdata["sliceable"]:
        return 3, f"{target_object_id} is not sliceable."
    if targ_obj_mdata["isSliced"]:
        return 3, f"{target_object_id} is already sliced."

    prev_llm_ids = AI2ThorEnv.obj_db.llm_ids

    # all preconditions satisfied. attempt to slice
    event = await AI2ThorEnv.step(
        agent_id, 
        action="SliceObject", 
        objectId=ai2thor_obj_id, 
        save_img=True, 
        iterate_steps=True
    )

    post_llm_ids = AI2ThorEnv.obj_db.llm_ids

    # get disjoint set between post and prev llm ids to get new ids
    new_llm_ids = list(post_llm_ids ^ prev_llm_ids)

    if not event.metadata["lastActionSuccess"]:
        return 3, f"Could not slice {target_object_id}."
    return 0, f"Sliced {target_object_id}. New objects discovered: {', '.join(new_llm_ids)}"
