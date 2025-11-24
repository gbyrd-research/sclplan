"""
Define the schemas for each action function for the AI2Thor environment.
"""

import os

from src.utils.classical_planner.pddl import PDDLActionArgMapping, ReactActionSchema
from src.utils.state.object_database import ObjectDatabase
from src.utils.environment.alfworld.action_functions import *

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)
ALFWORLD_PDDL_DOMAIN_PATH = os.path.join(CUR_DIR, "alfworld_domain.pddl")

################################################################################

# Examine Object In Light Action

search_for_object_arg_desc = {
    "target_object_type": "The non-plural object type of the object to look for.",
}

search_for_object_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="target_object_type",
        agent_state_var=None,
    )
]

search_for_object_schema = ReactActionSchema(
    action_function=search_for_object,
    llm_action_name="SearchForObject",
    llm_action_desc="Search for an unseen object in the environment. You can use this multiple times to find multiple objects of a specific type.",
    llm_arg_desc=search_for_object_arg_desc
)

################################################################################

# Examine Object In Light Action

examine_object_in_light_arg_desc = {
    "held_object_id": "The object to examine.",
    "light_source_object_id": "The light source that you wish to use to examine.",
}

def get_targ_light_source_receptacle(obj_db: ObjectDatabase, llm_action_input: dict):
    targ_light_source = llm_action_input["light_source_object_id"]
    # find the known receptacle of the target light source
    try:
        for mdata in obj_db.get_obj_metadata_list():
            if obj_db.ensure_env_id(mdata["objectId"]) == obj_db.ensure_env_id(targ_light_source):
                return obj_db.ensure_llm_id(mdata["inReceptacle"][1])
    except:
        return None

examine_object_in_light_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o",
        pddl_arg_type="object",
        llm_arg="held_object_id",
        agent_state_var=None,
    ),
    PDDLActionArgMapping(
        pddl_arg="?l",
        pddl_arg_type="object",
        llm_arg="light_source_object_id",
        agent_state_var=None,
    ),
    PDDLActionArgMapping(
        pddl_arg="?r", 
        pddl_arg_type="object", 
        llm_arg=None, 
        agent_state_var=None,
        obj_mdata_function=get_targ_light_source_receptacle
    ),
]

examine_object_in_light_schema = ReactActionSchema(
    action_function=examine_object_in_light,
    llm_action_name="ExamineObjectInLight",
    llm_action_desc="Examine an object in the light of a specified light source.",
    llm_arg_desc=examine_object_in_light_arg_desc,
    pddl_action_name="examineObjectInLight",
    pddl_action_arg_mappings=examine_object_in_light_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Go To Receptacle Action

go_to_receptacle_arg_desc = {
    "receptacle_id": "The receptacle to go to.",
}

go_to_receptacle_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?rEnd",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    )
]

go_to_receptacle_schema = ReactActionSchema(
    action_function=go_to_receptacle,
    llm_action_name="GoToReceptacle",
    llm_action_desc="Move to a target receptacle.",
    llm_arg_desc=go_to_receptacle_arg_desc,
    pddl_action_name="GotoReceptacle",
    pddl_action_arg_mappings=go_to_receptacle_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Open Receptacle Action

open_receptacle_arg_desc = {
    "receptacle_id": "The receptacle to open.",
}

open_receptacle_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    )
]

open_receptacle_schema = ReactActionSchema(
    action_function=open_receptacle,
    llm_action_name="OpenReceptacle",
    llm_action_desc="Open a receptacle.",
    llm_arg_desc=open_receptacle_arg_desc,
    pddl_action_name="OpenReceptacle",
    pddl_action_arg_mappings=open_receptacle_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Close Receptacle Action

close_receptacle_arg_desc = {
    "receptacle_id": "The receptacle to close.",
}

close_receptacle_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    )
]

close_receptacle_schema = ReactActionSchema(
    action_function=close_receptacle,
    llm_action_name="CloseReceptacle",
    llm_action_desc="Close a receptacle.",
    llm_arg_desc=close_receptacle_arg_desc,
    pddl_action_name="CloseReceptacle",
    pddl_action_arg_mappings=close_receptacle_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Pickup Object From Receptacle Action

pickup_object_from_receptacle_arg_desc = {
    "object_id": "The object to pick up.",
    "receptacle_id": "The receptacle to pick up the object from.",
}

pickup_object_from_receptacle_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o", pddl_arg_type="object", llm_arg="object_id", agent_state_var=None
    ),
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    ),
]

pickup_object_schema = ReactActionSchema(
    action_function=pickup_object_from_receptacle,
    llm_action_name="PickupObjectFromReceptacle",
    llm_action_desc="Pick up an object from a receptacle.",
    llm_arg_desc=pickup_object_from_receptacle_arg_desc,
    pddl_action_name="PickupObjectFromReceptacle",
    pddl_action_arg_mappings=pickup_object_from_receptacle_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Put Object Action

put_object_arg_desc = {
    "object_id": "The object to put down.",
    "receptacle_id": "The receptacle to put the object into or onto.",
}

put_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o", pddl_arg_type="object", llm_arg="object_id", agent_state_var=None
    ),
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    ),
]

put_object_schema = ReactActionSchema(
    action_function=put_object,
    llm_action_name="PutObject",
    llm_action_desc="Put down an object.",
    llm_arg_desc=put_object_arg_desc,
    pddl_action_name="PutObject",
    pddl_action_arg_mappings=put_object_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Clean Object Action

clean_object_arg_desc = {
    "object_id": "The object to clean.",
    "receptacle_id": "The receptacle to use to clean the object.",
}

clean_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o", pddl_arg_type="object", llm_arg="object_id", agent_state_var=None
    ),
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    ),
]

clean_object_schema = ReactActionSchema(
    action_function=clean_object,
    llm_action_name="CleanObject",
    llm_action_desc="Clean an object.",
    llm_arg_desc=clean_object_arg_desc,
    pddl_action_name="CleanObject",
    pddl_action_arg_mappings=clean_object_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Heat Object Action

heat_object_arg_desc = {
    "object_id": "The object to heat.",
    "receptacle_id": "The receptacle to use to heat the object.",
}

heat_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o", pddl_arg_type="object", llm_arg="object_id", agent_state_var=None
    ),
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    ),
]

heat_object_schema = ReactActionSchema(
    action_function=heat_object,
    llm_action_name="HeatObject",
    llm_action_desc="Heat an object.",
    llm_arg_desc=heat_object_arg_desc,
    pddl_action_name="HeatObject",
    pddl_action_arg_mappings=heat_object_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

################################################################################

# Cool Object Action

cool_object_arg_desc = {
    "object_id": "The object to cool.",
    "receptacle_id": "The receptacle to use to cool the object.",
}

cool_object_pddl_arg_mappings = [
    PDDLActionArgMapping(
        pddl_arg="?o", pddl_arg_type="object", llm_arg="object_id", agent_state_var=None
    ),
    PDDLActionArgMapping(
        pddl_arg="?r",
        pddl_arg_type="object",
        llm_arg="receptacle_id",
        agent_state_var=None,
    ),
]

cool_object_schema = ReactActionSchema(
    action_function=cool_object,
    llm_action_name="CoolObject",
    llm_action_desc="Cool an object.",
    llm_arg_desc=cool_object_arg_desc,
    pddl_action_name="CoolObject",
    pddl_action_arg_mappings=cool_object_pddl_arg_mappings,
    pddl_domain_path=ALFWORLD_PDDL_DOMAIN_PATH,
)

ALFWORLD_ACTION_SCHEMAS_0 = [
    search_for_object_schema,
    examine_object_in_light_schema,
    go_to_receptacle_schema,
    open_receptacle_schema,
    close_receptacle_schema,
    pickup_object_schema,
    put_object_schema,
    clean_object_schema,
    heat_object_schema,
    cool_object_schema,
]
