"""
Define the schemas for each action function for the AI2Thor environment.
"""

import os

from src.utils.classical_planner.pddl import PDDLActionArgMapping, ReactActionSchema
from src.utils.environment.ai2thor.action_functions import (
    move_to_object,
    open_object,
    close_object,
    pickup_object,
    place_object,
    slice_object,
    toggle_object_off,
    toggle_object_on,
)

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)
AI2THOR_PDDL_DOMAIN_PATH = os.path.join(CUR_DIR, "ai2thor_domain_0.pddl")

################################################################################

# Move To Object Action

move_to_object_arg_desc = {"object_id": "The object id of the object to move to."}

move_to_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?lStart",
        pddl_arg_type="location",
        llm_arg=None,
        agent_state_var="cur_loc",
    ),
    PDDLActionArgMapping(
        pddl_arg="?lEnd",
        pddl_arg_type="location",
        llm_arg="object_id",
        agent_state_var=None,
    ),
]

move_to_object_schema = ReactActionSchema(
    action_function=move_to_object,
    llm_action_name="MoveToObject",
    llm_action_desc="Move to an object.",
    llm_arg_desc=move_to_object_arg_desc,
    pddl_action_name="MoveAgentToLocation",
    pddl_action_arg_mappings=move_to_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Open Object Action

open_object_arg_desc = {"object_id": "The object id of the object to open."}

open_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="object_id",
        agent_state_var=None,
    ),
]

open_object_schema = ReactActionSchema(
    action_function=open_object,
    llm_action_name="OpenObject",
    llm_action_desc="Open an object.",
    llm_arg_desc=open_object_arg_desc,
    pddl_action_name="OpenObject",
    pddl_action_arg_mappings=open_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Close Object Action

close_object_arg_desc = {"object_id": "The object id of the object to close."}

close_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="object_id",
        agent_state_var=None,
    ),
]

close_object_schema = ReactActionSchema(
    action_function=close_object,
    llm_action_name="CloseObject",
    llm_action_desc="Close an object.",
    llm_arg_desc=close_object_arg_desc,
    pddl_action_name="CloseObject",
    pddl_action_arg_mappings=close_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Pickup Object Action

pickup_object_arg_desc = {"object_id": "The object id of the object to pick up."}

pickup_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="object_id",
        agent_state_var=None,
    ),
]

pickup_object_schema = ReactActionSchema(
    action_function=pickup_object,
    llm_action_name="PickupObject",
    llm_action_desc="Pickup an object.",
    llm_arg_desc=pickup_object_arg_desc,
    pddl_action_name="PickUpObject",
    pddl_action_arg_mappings=pickup_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Place Object Action

place_object_arg_desc = {
    "held_object_id": "The object id of the held object to place",
    "receptacle_id": "The object id of the target object to place the held object on top of or inside.",
}

place_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?heldObject",
        pddl_arg_type="object",
        llm_arg="held_object_id",
        agent_state_var=None,
    ),
    PDDLActionArgMapping(
        pddl_arg="?targetReceptacle",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    ),
]

place_object_schema = ReactActionSchema(
    action_function=place_object,
    llm_action_name="PlaceObject",
    llm_action_desc="Place a held object on top of or inside a target receptacle.",
    llm_arg_desc=place_object_arg_desc,
    pddl_action_name="PlaceObject",
    pddl_action_arg_mappings=place_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Toggle Object On Action

toggle_on_object_arg_desc = {"object_id": "The object id of the object to toggle on."}

toggle_on_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="object_id",
        agent_state_var=None,
    ),
]

toggle_on_object_schema = ReactActionSchema(
    action_function=toggle_object_on,
    llm_action_name="ToggleObjectOn",
    llm_action_desc="Toggle an object on.",
    llm_arg_desc=toggle_on_object_arg_desc,
    pddl_action_name="ToggleObjectOn",
    pddl_action_arg_mappings=toggle_on_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Toggle Object Off Action

toggle_off_object_arg_desc = {"object_id": "The object id of the object to toggle off."}

toggle_off_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="object_id",
        agent_state_var=None,
    ),
]

toggle_off_object_schema = ReactActionSchema(
    action_function=toggle_object_off,
    llm_action_name="ToggleObjectOff",
    llm_action_desc="Toggle an object off.",
    llm_arg_desc=toggle_off_object_arg_desc,
    pddl_action_name="ToggleObjectOff",
    pddl_action_arg_mappings=toggle_off_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

################################################################################

# Slice Object Action

slice_object_arg_desc = {
    "slicing_object_id": "The object id of the object you will use as a tool to slice the target object.",
    "target_object_id": "The object id of the object to cut into slices.",
}

slice_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?a",
        pddl_arg_type="agent",
        llm_arg=None,
        agent_state_var="agent_id_str",
    ),
    PDDLActionArgMapping(
        pddl_arg="?heldObject",
        pddl_arg_type="object",
        llm_arg="slicing_object_id",
        agent_state_var=None,
    ),
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="target_object_id",
        agent_state_var=None,
    ),
]

slice_object_schema = ReactActionSchema(
    action_function=slice_object,
    llm_action_name="SliceObject",
    llm_action_desc="Cut an object into slices.",
    llm_arg_desc=slice_object_arg_desc,
    pddl_action_name="SliceObject",
    pddl_action_arg_mappings=slice_object_pddl_arg_mappings,
    pddl_domain_path=AI2THOR_PDDL_DOMAIN_PATH,
)

AI2THOR_ACTION_SCHEMAS_0 = [
    move_to_object_schema,
    open_object_schema,
    close_object_schema,
    pickup_object_schema,
    place_object_schema,
    toggle_on_object_schema,
    toggle_off_object_schema,
    slice_object_schema,
]
