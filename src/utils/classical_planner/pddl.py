import glob
import inspect
import os
import shutil
import subprocess
import warnings
from typing import Callable, Dict, List, Optional, Tuple
import re
import inspect
import platform

from src.utils.agent.action import AgentAction
from src.utils.misc import logging_utils
from src.utils.state.agent_state import AgentState, ObjectDatabase

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)
PLANNER = os.path.join(CUR_DIR, "fast-downward-22.12/fast-downward.py")


class PDDLActionArgMapping:
    """Use this class to define mappings from PDDL arguments to LLM arguments
    or agent state variables in a formalized way."""

    def __init__(
        self,
        pddl_arg: str,
        pddl_arg_type: str,
        llm_arg: Optional[str] = None,
        agent_state_var: Optional[str] = None,
        obj_mdata_function: Callable = None
    ):
        """The obj_mdata_function will recieve two parameters: 
            1. the llm input to the function in dictionary format
            2. the entire object database
            
        It will then return a value based on that. This allows for more complex
        user defined functions.
        """
        llm_arg_check = int(llm_arg is not None)
        agent_state_var_check = int(agent_state_var is not None)
        obj_mdata_lambda_function_check = int(obj_mdata_function is not None)

        if llm_arg_check + agent_state_var_check + obj_mdata_lambda_function_check != 1:
            raise Exception("Must define ONE of llm_arg or agent_state_var or obj_mdata_function.")

        self.pddl_arg = pddl_arg
        self.pddl_arg_type = pddl_arg_type
        self.llm_arg = llm_arg
        self.agent_state_var = agent_state_var

        # do some verification on the format of the obj_mdata_function
        if obj_mdata_function is not None:
            required_params = {
                "obj_db": ObjectDatabase, 
                "llm_action_input": dict
            }
            sig = inspect.signature(obj_mdata_function)
            params = sig.parameters
            for param_name, expected_type in required_params.items():
                # Ensure parameter exists
                if param_name not in params:
                    raise Exception(f"obj_mdata_function must include {param_name} parameter.")

                # Ensure correct type hint
                param = params[param_name]
                if param.annotation is param.empty or param.annotation is not expected_type:
                    raise Exception(f"obj_mdata_function param {param_name} must be of type {expected_type}")

        self.obj_mdata_function = obj_mdata_function


import inspect

def check_function_params(func, required_params):
    """
    Checks if the given function has the required keyword parameters with specified types.

    :param func: The function to check.
    :param required_params: A dictionary with parameter names as keys and expected types as values.
    :return: True if the function meets the criteria, otherwise False.
    """
    # Get function signature
    sig = inspect.signature(func)

    # Extract parameters
    params = sig.parameters

    for param_name, expected_type in required_params.items():
        # Ensure parameter exists
        if param_name not in params:
            return False
        
        # Ensure it is a keyword parameter
        param = params[param_name]
        if param.default is param.empty:
            return False  # Not a keyword parameter
        
        # Ensure correct type hint
        if param.annotation is param.empty or param.annotation is not expected_type:
            return False
    
    return True

class ReactActionSchema:
    """Use this class to define a action to pass to ReAct agent."""

    def __init__(
        self,
        action_function: Callable,
        llm_action_name: str,
        llm_action_desc: str,
        llm_arg_desc: Dict[str, str],
        pddl_action_name: Optional[str] = None,
        pddl_action_arg_mappings: Optional[List[PDDLActionArgMapping]] = None,
        pddl_domain_path: Optional[str] = None,
    ):
        """Defines a action to pass to a ReAct agent.

        Args:
            action_function (Callable): the actual action function that will be
                called
            llm_action_name (str): the name of the action that will be provided
                to the LLM
            llm_action_desc (str): a description of what the action does to be passed
                to the llm
            llm_arg_desc (Dict[str, str]): for each argument in the action function
                provide a mapping from the name of that argument to a description
                that will be provided to the LLM
            pddl_action_name (Optional[str]): if using a classical planner, you must
                provide the corresponding pddl action name
            pddl_action_arg_mappings (Optional[List[PDDLActionArgMapping]]): if using
                a classical planner, each pddl action must have parameters that
                can be mapped to either an argument to the python action called by
                the llm or an attribute of the agent state. this argument provides
                a list of such mappings

        Returns:
            None
        """
        self.action_function = action_function
        self.llm_action_name = llm_action_name
        self.llm_action_desc = llm_action_desc
        self.llm_arg_desc = llm_arg_desc
        self.pddl_action_name = pddl_action_name
        self.pddl_action_arg_mappings = pddl_action_arg_mappings
        self.pddl_domain_path = pddl_domain_path

    def _verify(self) -> None:
        """Verify that everything matches between the LLM definition, action function
        call and the corresponding pddl action.

        Args:
            None

        Returns:
            None
        """
        # if any pddl attributes were devined, ensure that all were provided
        # and none were left with None value
        pddl_args = (
            self.pddl_action_name,
            self.pddl_action_arg_mappings,
            self.pddl_domain_path,
        )
        if not (
            all(v is None for v in pddl_args) or all(v is not None for v in pddl_args)
        ):
            raise Exception(
                "Some pddl attributes were defined while others left as None. All must be defined if one is defined."
            )

        # if pddl values were defined
        if all(v is not None for v in pddl_args):
            assert self.pddl_action_arg_mappings is not None
            assert self.pddl_action_name is not None
            assert self.pddl_domain_path is not None

            # verify that the pddl action block exists
            action_block_substring = f"(action {self.pddl_action_name}"
            action_block = get_pddl_block_from_pddl_file(
                self.pddl_domain_path, action_block_substring
            )

            # verify that the pddl parameters block exists
            params = get_subblock_from_action_block_str(action_block, ":parameters")
            params = parse_arg_names_and_types_from_params_block(params)

            # verify the specified pddl action arg mappings
            for m in self.pddl_action_arg_mappings:
                if m.pddl_arg not in params:
                    raise Exception(
                        "Specified pddl action arguments does not match domain file."
                    )
                if m.pddl_arg_type != params[m.pddl_arg]:
                    raise Exception(
                        "Specified pddl action arg type does not match domain file."
                    )
                if m.llm_arg is not None and m.llm_arg not in self.llm_arg_desc:
                    raise Exception(
                        "Specified llm arg corresponding to pddl arg does not match any defined llm args. Please ensure correspondence."
                    )
                if m.agent_state_var is not None and not hasattr(
                    AgentState, m.agent_state_var
                ):
                    raise Exception(
                        "The specified agent state var corresponding to pddl arg is not a valid attribute of the AgentState."
                    )
        # verify the llm facing action arguments match those of the action
        # action function
        action_sig = inspect.signature(self.action_function)
        actual_action_args_str_list = list(action_sig.parameters.keys())

        for llm_facing_arg in self.llm_arg_desc:
            if llm_facing_arg not in actual_action_args_str_list:
                raise Exception(
                    "An llm facing arg MUST match an arg in the actual action function."
                )


# TODO: Some of these methods are hardcoded to ai2thor..
# This needs to be addressed if we hope to make this modular
# to different environments, but that is not currently priority


# # these are the object boolean attributes that we need to create
# # predicates for
# # NOTE: we do not include the receptacle here an object
# # will be defined as type "receptacle" if it is a receptacle, therefore
# # we do not need a predicate to keep track of this
# relevant_obj_bool_atts = {
#     "visible",
#     "isInteractable",
#     "toggleable",
#     "isToggled",
#     "breakable",
#     "isBroken",
#     "canFillWithLiquid",
#     "isFilledWithLiquid",
#     "dirtyable",
#     "isDirty",
#     "canBeUsedUp",
#     "isUsedUp",
#     "cookable",
#     "isCooked",
#     "isHeatSource",
#     "isColdSource",
#     "sliceable",
#     "isSliced",
#     "openable",
#     "isOpen",
#     "pickupable",
#     "isPickedUp",
#     "moveable",
# }

def get_predicate_names_from_predicates_block(predicates_block: str) -> List[str]:
    predicate_pattern = re.compile(r'\(\s*(\w+)')

    # Find all matches in the PDDL block
    matches = predicate_pattern.findall(predicates_block)

    return matches

def create_pddl_problem_file(
    goal_block: List[str] | str, 
    save_path: str, 
    domain_path: str, env
) -> None:
    """

    Args:
        goal_block (List[str]): a list of strings that when saved to a
            file using file.writelines() would save a properly formatted
            pddl :goal block
        save_path (str): the save path of the problem.pddl file

    Returns:
        None
    """
    # parse the domain file to get the domain name
    with open(domain_path, "r") as file:
        domain = file.readline()
    domain_name = domain.split(" ")[-1][:-2]

    # get state of environment (discovered objects)
    obj_db = env.obj_db
    agent_states = env.agent_states
    agent_ids = list(agent_states.keys())

    # get the problem.pddl :objects and :init blocks
    if "ai2thor" in env.__class__.__name__.lower():
        objects_block = get_pddl_objects_block_ai2thor(obj_db, [f"agent_{id}" for id in agent_ids])
        # create the :init predicates representing the agent state
        agent_state_pddl = list()
        for a_state in agent_states.values():
            agent_state_pddl.extend(a_state.get_state_in_pddl_format(obj_db))
    elif "alfworld" in env.__class__.__name__.lower():
        objects_block = get_pddl_objects_block_alfworld(obj_db, [f"agent_{id}" for id in agent_ids])
        agent_state_pddl = [] # there is no agent state for the alfworld domain
    else:
        raise NotImplementedError("Parsing an objects block for this environment has not yet been implemented.")

    predicates_block = get_pddl_block_from_pddl_file(domain_path, "(:predicates")
    valid_atts = get_predicate_names_from_predicates_block(predicates_block)

    init_block = get_pddl_init_block(obj_db, agent_state_pddl, valid_atts, env)

    # create the problem.pddl file
    with open(save_path, "w") as file:
        # write the first `define` line of the problem.pddl file
        l1 = f"(define (problem temp_problem) (:domain {domain_name})\n"
        file.write(l1)

        # write the :objects block
        file.writelines(objects_block)

        # write the :init block
        file.writelines(init_block)

        # write the :goal block
        file.writelines(goal_block)

        # close the open parenthesis from the `define` line (line 1)
        file.write(")")


def parse_obj_mdata_to_pddl_init_block_ai2thor(
    llm_id: str, 
    obj_mdata: Dict, 
    obj_db: ObjectDatabase,
    valid_obj_atts: List[str]
) -> List[str]:
    """Takes in a dictionary of object metadata from the ai2thor
    simulator and parses the metadata into a valid state initialization
    for that object in the form of a list of strings.

    Args:
        obj_mdata (Dict): a dictionary representing an object's metadata

    Returns:
        pddl_starting_state (List[str]): a list of strings that when
            added to a pddl :init block represent the object's starting
            state.
    """
    init_state_list = list()
    for att in valid_obj_atts:
        pddl_starting_state = f"\t({att} {llm_id})"
        if att in obj_mdata:
            if isinstance(obj_mdata[att], bool):
                if obj_mdata[att]:
                    init_state_list.append(pddl_starting_state + "\n")
                else:
                    init_state_list.append(f"(not {pddl_starting_state})\n")


    # add the state of objects inside of other objects
    parent_receptacles = obj_mdata["parentReceptacles"]
    if parent_receptacles is not None and len(parent_receptacles) > 0:
        for parent_receptacle in parent_receptacles:
            if parent_receptacle not in obj_db.env_ids:
                warnings.warn(
                    "Parent receptacle of discovered object not yet in object database."
                )
            init_state_list.append(
                f"(objectInReceptacle {obj_db.ensure_llm_id(llm_id)} {obj_db.ensure_llm_id(parent_receptacle)})\n"
            )

    # add the isValidReceptacles predicates for each item
    if "valid_receptacles" in obj_mdata:
        for valid_receptacle in obj_mdata["valid_receptacles"]:
            init_state_list.append(
                f"(isValidReceptacle {obj_db.ensure_llm_id(llm_id)} {obj_db.ensure_llm_id(valid_receptacle)})\n"
            )

    # add special attributes that are not inherently provided by ai2thor
    # object metadata
    if "knife" in llm_id.lower():
        init_state_list.append(f"\t(isKnife {llm_id})\n")
    return init_state_list

def parse_obj_mdata_to_pddl_init_block_alfworld(
    llm_id: str, 
    obj_mdata: Dict, 
    obj_db: ObjectDatabase,
    valid_obj_atts: List[str]
) -> List[str]:
    """Takes in a dictionary of object metadata from the ai2thor
    simulator and parses the metadata into a valid state initialization
    for that object in the form of a list of strings.

    Args:
        obj_mdata (Dict): a dictionary representing an object's metadata

    Returns:
        pddl_starting_state (List[str]): a list of strings that when
            added to a pddl :init block represent the object's starting
            state.
    """
    init_state_list = list()
    for att in valid_obj_atts:
        # there are two cases currently, one for boolean attributes and one
        # for predicates that take multiple "arguments" which will be provided
        # in a list where the order of the list is the same as the order of
        # "arguments" after the predicate
        if att in obj_mdata:
            if isinstance(obj_mdata[att], bool):
                pddl_starting_state = f"\t({att} {obj_db.ensure_llm_id(llm_id)})"
                if obj_mdata[att]:
                    init_state_list.append(pddl_starting_state + "\n")
                # else:
                #     init_state_list.append(f"(not {pddl_starting_state})\n")
            elif isinstance(obj_mdata[att], list):
                input_values = list()
                for x in obj_mdata[att]:
                    if obj_db.check_if_id_exists(str(x)):
                        input_values.append(obj_db.ensure_llm_id(str(x)))
                    else:
                        input_values.append(str(x))
                pddl_starting_state = f"\t({att} {' '.join(input_values)})\n"
                init_state_list.append(pddl_starting_state)
            
    return init_state_list


def get_pddl_init_block(
    obj_db: ObjectDatabase, agent_state_init: List[str], valid_obj_atts: List[str], env
) -> List[str]:
    """Takes in a list of object metadata dictionaries and a list of predicates
    representing each agent's starting state and returns a list of strings
    corresponding the :init block of a .pddl problem file.

    Args:
        obj_metadata_list: List[Dict]: a list of object metadata
            dictionaries.
        agent_starting_state: List[str]: a list of strings representing the
            starting state of agents

    Returns:
        pddl_init_block (List[str]): a list of strings that when
            written to a .pddl file with
            file.writelines(pddl_init_block) creates the :init
            block of a .pddl problem file.
    """

    # generate the :init block
    init_block = list()
    init_block.append("(:init\n")
    obj_metadata_list = obj_db.get_obj_metadata_list()
    for mdata in obj_metadata_list:
        llm_id = obj_db.env_to_llm(mdata["objectId"])
        if "ai2thor" in env.__class__.__name__.lower():
            init_block.extend(parse_obj_mdata_to_pddl_init_block_ai2thor(llm_id, mdata, obj_db, valid_obj_atts))
        elif "alfworld" in env.__class__.__name__.lower():
            init_block.extend(parse_obj_mdata_to_pddl_init_block_alfworld(llm_id, mdata, obj_db, valid_obj_atts))
    init_block.extend(["\t" + x for x in agent_state_init])
    init_block.append(")\n")

    return init_block


def get_pddl_objects_block_ai2thor(obj_db: ObjectDatabase, agents_list: List[str]) -> List[str]:
    """Takes in a list of object metadata dictionaries and returns
    a list of strings corresponding the :objects block of a .pddl
    problem file.

    Args:
        obj_metadata_list: List[Dict]: a list of object metadata
            dictionaries.
        agents_list: List[Dict]: a list of strings representing agents
            in the format agent_<agent_id> e.g. agent_1

    Returns:
        pddl_objects_block (List[str]): a list of strings that when
            written to a .pddl file with
            file.writelines(pddl_objects_block) creates the :objects
            block of a .pddl problem file.
    """
    # keep track of plain objects and receptacles with a list of their
    # object ids
    obj_types = {}

    obj_metadata_list = obj_db.get_obj_metadata_list()

    for mdata in obj_metadata_list:
        env_id = mdata["objectId"]
        obj_type = mdata["objectType"].lower()
        # quick hack because we do not want two different types for knife and
        # butterknife
        if "knife" in obj_type:
            obj_type = "knife"
        llm_id = obj_db.env_to_llm(env_id)
        if obj_type not in obj_types:
            obj_types[obj_type] = list()
        obj_types[obj_type].append(llm_id)

    obj_types["agent"] = list()
    for agent in agents_list:
        obj_types["agent"].append(agent)

    # generate the :objects block
    objects_block = list()
    objects_block.append("(:objects\n")
    for t in obj_types:
        objects_block.append("\t" + " ".join(obj_types[t]) + f" - {t}\n")
    objects_block.append(")\n")

    return objects_block

def get_pddl_objects_block_alfworld(obj_db: ObjectDatabase, agents_list: List[str]) -> List[str]:
    """Takes in a list of object metadata dictionaries and returns
    a list of strings corresponding the :objects block of a .pddl
    problem file.

    Args:
        obj_metadata_list: List[Dict]: a list of object metadata
            dictionaries.
        agents_list: List[Dict]: a list of strings representing agents
            in the format agent_<agent_id> e.g. agent_1

    Returns:
        pddl_objects_block (List[str]): a list of strings that when
            written to a .pddl file with
            file.writelines(pddl_objects_block) creates the :objects
            block of a .pddl problem file.
    """
    # keep track of plain objects and receptacles with a list of their
    # object ids
    obj_types = {}

    obj_metadata_list = obj_db.get_obj_metadata_list()

    for mdata in obj_metadata_list:
        env_id = mdata["objectId"]
        obj_type = mdata["objectType"]
        llm_id = obj_db.env_to_llm(env_id)
        if obj_type not in obj_types:
            obj_types[obj_type] = list()
        obj_types[obj_type].append(llm_id)

    # generate the :objects block
    objects_block = list()
    objects_block.append("(:objects\n")
    for t in obj_types:
        objects_block.append("\t" + " ".join(obj_types[t]) + f" - {t}\n")
    objects_block.append(")\n")

    return objects_block


def get_pddl_block_from_pddl_file(pddl_file_path: str, block_identifier: str) -> str:
    """Parses a pddl file for a block specified by some identifier.

    For example, (:goal will get the goal block and (:action my_action_1
    will get the action block corresponding to an action named my_action_1.

    Args:
        pddl_file_path (str): Path to the .pddl file.
        block_identifier (str): Substring representing

    Returns:
        str: The found block as a string.
    """
    try:
        with open(pddl_file_path, "r") as file:
            pddl_file_data = file.read()

        start_idx = pddl_file_data.find(block_identifier)
        if start_idx == -1:
            raise ValueError(
                f"The substring {block_identifier} not found in the pddl action block string."
            )

        pddl_file_data = pddl_file_data[start_idx:]
        open_paren_count = 1
        closed_paren_count = 0
        end_idx = pddl_file_data.find("(") + 1
        while open_paren_count != closed_paren_count:
            if pddl_file_data[end_idx] == "(":
                open_paren_count += 1
                end_idx += 1
                continue
            if pddl_file_data[end_idx] == ")":
                closed_paren_count += 1
                end_idx += 1
                continue
            end_idx += 1
            if end_idx == 2000000:
                raise Exception(
                    "Closing bracket not found. Make sure block contains closing bracket."
                )
        return pddl_file_data[:end_idx]

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{pddl_file_path}' does not exist.")
    except Exception as e:
        raise e


def get_subblock_from_action_block_str(
    action_block_str: str, subblock_id_str: str
) -> str:
    """
    Extracts and returns either the :parameters, :precondition, or :effect
    subblock from the action block str.

    Args:
        action_block_str (str): the string representing the action block
        subblock_id_str (str): the string representing the subblock you want to
            find

    Returns:
        subblock (str): the subblock that starts after the subblock_id_str
            in the action_block_str
    """
    valid_subblock_ids = [":parameters", ":precondition", ":effect"]
    if subblock_id_str not in valid_subblock_ids:
        raise ValueError(
            f"{subblock_id_str} invalid. Must be one of {valid_subblock_ids}."
        )

    start_idx = action_block_str.find(subblock_id_str)
    if start_idx == -1:
        raise ValueError(
            f"The substring {subblock_id_str} not found in the pddl action block string."
        )

    action_block_str = action_block_str[start_idx:]
    open_paren_count = 1
    closed_paren_count = 0
    end_idx = action_block_str.find("(") + 1
    while open_paren_count != closed_paren_count:
        if action_block_str[end_idx] == "(":
            open_paren_count += 1
            end_idx += 1
            continue
        if action_block_str[end_idx] == ")":
            closed_paren_count += 1
            end_idx += 1
            continue
        end_idx += 1
        if end_idx == 2000:
            raise Exception(
                "Closing bracket not found. Make sure block contains closing bracket."
            )
    return action_block_str[:end_idx]


def parse_arg_names_and_types_from_params_block(param_block_str: str) -> Dict:
    """Takes in a parameters block and parses out the argument names
    and types to a dictionary that is returned."""
    params = dict()

    temp = param_block_str.replace("(", "")
    temp = temp.replace(")", "")
    temp = temp.replace(" ", "")
    temp = temp.split("?")[1:]
    for i in temp:
        l = i.split("-")
        params[f"?{l[0]}"] = l[1]

    return params


def get_pddl_action_name(
    llm_action_name: str,
    actions,  #: List[ReactActionSchema]
) -> str:
    action_schema = [a for a in actions if a.llm_action_name == llm_action_name][0]
    pddl_action_name = action_schema.pddl_action_name
    if pddl_action_name is None:
        # raise Exception(
        #     f"PDDL action name not defined for llm action name: {llm_action_name}"
        # )
        return None
    return pddl_action_name


def get_pddl_action_input(
    llm_action_name: str,
    llm_action_input: Dict,
    actions,  #: List[ReactActionSchema],
    state: AgentState,
    env,
) -> Dict:
    action_schema = [a for a in actions if a.llm_action_name == llm_action_name][0]
    pddl_mappings_list = action_schema.pddl_action_arg_mappings
    if pddl_mappings_list is None:
        # raise Exception(
        #     f"PDDL action arg mappings not defined for llm action name: {llm_action_name}."
        # )
        return None

    pddl_input = dict()
    for p in pddl_mappings_list:
        if p.llm_arg is not None:
            pddl_input[p.pddl_arg] = llm_action_input[p.llm_arg]

            # the llm argument presents object ids in LLM facing format
            # whereas the pddl object id presents in ai2thor format. we
            # will convert here if necessary
            if pddl_input[p.pddl_arg] in env.obj_db.llm_ids.union(
                env.obj_db.env_ids
            ):
                pddl_input[p.pddl_arg] = env.obj_db.ensure_llm_id(
                    pddl_input[p.pddl_arg]
                )

        elif p.agent_state_var is not None:
            if not hasattr(state, p.agent_state_var):
                raise Exception(f"Agent state has no attribute {p.agent_state_var}.")
            pddl_input[p.pddl_arg] = getattr(state, p.agent_state_var)
        elif p.obj_mdata_function is not None:
            pddl_input[p.pddl_arg] = p.obj_mdata_function(env.obj_db, llm_action_input)
        else:
            raise Exception(
                "No pddl mapping defined for either an llm argument or agent state variable or object metadata function."
            )
    return pddl_input


def get_goal_state_from_preconditions(
    agent_action: AgentAction, pddl_domain_file_path: str
):
    """Takes in an AgentAction object, parses the corresponding pddl
    action name from the AgentAction, uses this name to parse the preconditions
    for the corresponding defined pddl action in the pddl domain file. Finally,
    uses these preconditions to define a goal state in which the preconditions
    would be satisfied.

    Args:
        agent_action (AgentAction): the agent action object that defines the
            action whose preconditions you wish to provide a goal state for
        pddl_domain_file_path (str): the path to the domain.pddl file
    """

    pddl_action_name = agent_action.pddl_action_name
    pddl_action_input = agent_action.pddl_action_input

    # get the preconditions from the pddl domain action block
    action_block = get_pddl_block_from_pddl_file(
        pddl_domain_file_path, f"(:action {pddl_action_name}"
    )
    preconditions = get_subblock_from_action_block_str(action_block, ":precondition")

    # replace the precondition arguments with the actual values from the
    # agent_actions's pddl_action_input
    if pddl_action_input is not None:
        for pddl_arg, pddl_val in pddl_action_input.items():
            if pddl_val is None:
                pddl_val = "n/a"
            preconditions = preconditions.replace(pddl_arg, pddl_val)

    goal_block = preconditions.replace(":precondition", ":goal")
    goal_block = "(" + goal_block + ")"

    return goal_block


def check_if_goal_state_satisfied(pddl_problem_file_path: str, domain_file_path: str, solver: str):
    """Takes in a pddl problem file path and determines if the goal
    state defined in the problem file is satisfied based on the state
    of the environment defined in the problem file.
    """
    # env_state = get_pddl_block_from_pddl_file(pddl_problem_file_path, "(:init")
    # goal_state = get_pddl_block_from_pddl_file(pddl_problem_file_path, "(:goal")

    temp_verification_dir = os.path.dirname(pddl_problem_file_path)
    logging_utils.ensure_dir_exists(temp_verification_dir)
    temp_plan_path = os.path.join(temp_verification_dir, "plan.txt")

    if solver == "lapkt":
        return verify_predicates_lapkt_solver(temp_plan_path, domain_file_path, pddl_problem_file_path, solver)

    result = create_plan(temp_plan_path, domain_file_path, pddl_problem_file_path, solver)

    # enact the plan to attempt to reach a state where all preconditions
    # are satisfied
    plan_lines = []
    if os.path.exists(temp_plan_path):
        with open(temp_plan_path, "r") as file:
            plan_lines = file.readlines()
    # else:
    #     # clean up the temp dirs and files
    #     # shutil.rmtree(temp_verification_dir)
    #     return True
    if result != 0:
        return False

    if (
        result == 0  # a plan.txt file was generated
        and ((len(plan_lines) > 1 and solver == "fast_downward")
             or (len(plan_lines) > 0 and solver == "lapkt"))
          # if this is true, plan file contains actions
    ):
        return False
    else:
        return True


def create_plan(
    output_path: str,
    pddl_domain_path: str,
    pddl_problem_path: str,
    solver: str,
    timeout: float = 5.0,
) -> int:
    """Generate a plan for the given pddl domain and problem files using
    the classical planner. Output the plan to the specifed output path.

    Args:
        output_path (str): the output path for the generated plan.txt file
        pddl_domain_path (str): the path the the pddl domain file for the planner
        pddl_problem_path (str): the path to the pddl problem file for the planner
        timeout (float): specify the max number of seconds for planning before
            timeout

    Returns:
        flag (int): specifies the result of the planner
            0: valid plan found
            1: valid plan not found due to timeout
            2: valid plan not found after fully searched state-action space
    """
    valid_solver = ["fast_downward", "lapkt"]
    if solver not in valid_solver:
        raise ValueError(f"Specified solver ({solver}) invalid. Must be one of {', '.join(valid_solver)}.")
    
    if solver == "fast_downward":
        # the Fast Downward planner runs as an executable, so we will need to spin
        # up a subprocess in order to run the plan. below we will define the
        # arguments for said subprocess

        # I use this to turn off my auto-formatter - Grayson
        # fmt: off
        command = [
            PLANNER,                        # path to the planner executable
            "--plan-file", output_path,     # path to the plan output file #ignore
            "--alias", "seq-sat-lama-2011",  # sets a pre-written configuration for the planner
            pddl_domain_path,               # path to the domain.pddl file
            pddl_problem_path               # path to the problem.pddl file
        ]
        # fmt on

        # NOTE: for more information on the arguments for Fast Downward, find
        # your fast-downward.py file and run:
        #   fast-downward.py --help

        # when using an alias, you cannot easily set the time limit for planning,
        # so I hack this functionality be setting a timemout for the spawned
        # process
        did_planner_finish = False
        try:
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=timeout # sets a timeout to limit the planner
            )
            did_planner_finish = True
        except Exception as e:
            # planner timed out while searching for optimal plan, although
            # suboptimal plans may have been generated
            # warnings.warn("Planner timed out before finding optimal plan...")
            pass

        """The planner will often generate a suboptimal plan, continue to search
        for a better plan, and then save another plan if a more optimal one is
        found. For example, plan.txt.1 will be saved. If a better plan is found,
        plan.txt.2 will be saved. We want to ensure that the best plan is saved
        out to the specified name in our function arguments (i.e. the output_path
        variable). to do this, we will search through all generated plans, rename 
        the best plan to the specified output_path name, and move all suboptimal
        plans to a separate folder for record keeping."""

        # find all generated plans
        abs_path = os.path.abspath(output_path)
        dirname = os.path.dirname(abs_path)
        saved_plans = list()
        output_fn = output_path.split("/")[-1]
        for fp in glob.glob(os.path.join(dirname, "*")):
            if os.path.isfile(fp) and output_fn in fp:
                saved_plans.append(fp)
        
        if len(saved_plans) == 0:
            # no plan was found
            if did_planner_finish:
                return 2 # no possible plan found after fully searched space
            else:
                return 1 # timeout. planner could not find a valid plan in time
        
        # sort by integer suffix
        saved_plans = sorted(saved_plans, key=lambda x: int(os.path.split(x)[-1].split(".")[-1]))

        # rename the best plan
        best_plan_path = saved_plans[-1] # best plan is plan with largest integer
        os.rename(best_plan_path, output_path)

        # create a folder for the suboptimal plans and place each suboptimal plan
        # in the folder
        suboptimal_plan_folder = os.path.join(dirname, "suboptimal_plans")
        if not os.path.exists(suboptimal_plan_folder):
            os.makedirs(suboptimal_plan_folder)
        for suboptimal_plan in saved_plans[:-1]:
            fn = os.path.split(suboptimal_plan)[-1]
            os.rename(suboptimal_plan, os.path.join(suboptimal_plan_folder, fn))

        return 0 # valid plan generated
    
    elif solver == "lapkt":
        pddl_dir = os.path.dirname(pddl_problem_path)

        if os.path.dirname(pddl_domain_path) != pddl_dir:
            new_temp_domain_path = os.path.join(pddl_dir, os.path.basename(pddl_domain_path))
            shutil.copy(pddl_domain_path, new_temp_domain_path)
            pddl_domain_path = new_temp_domain_path

        if (os.path.dirname(pddl_domain_path) != os.path.dirname(pddl_problem_path)
            or os.path.dirname(pddl_domain_path) != os.path.dirname(output_path)):
            raise Exception("All pddl files must be in the same directory due to how the files are copied into the docker container used for the solver.")

        # create the plan file that the planner will write to
        output_file = os.path.basename(output_path)

        # example command:
        # docker run --platform linux/amd64 --rm
        # -v ~/gpt-planner:/data lapkt/lapkt-public
        # timeout 30s ./ff --domain /data/alfworld_domain.pddl
        #                 --problem /data/out_problem.pddl
        #                 --output /data/plan.ipc

        arch = platform.machine().lower()
        if arch in ["x86_64", "amd64"]:
            docker_repo = "lapkt/lapkt-public"
        elif arch in ["arm64", "aarch64"]:
            docker_repo = "gautierdag/lapkt-arm"

        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.abspath(pddl_dir)}:/data",
            f"{docker_repo}",
            "timeout",
            f"{timeout}s",
            f"./bfs_f",
            "--domain",
            f"/data/{os.path.basename(pddl_domain_path)}",
            "--problem",
            f"/data/{os.path.basename(pddl_problem_path)}",
            "--output",
            f"/data/{os.path.basename(output_file)}"
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # if result.returncode != 0:
                # logger.warning(f"Command failed with return code {result.returncode}")
                # logger.warning(f"Standard output: {result.stdout.decode()}")
                # logger.error(f"Standard error: {result.stderr.decode()}")
                # logger.error("Planner Failure")
                # return 1
            if os.path.exists(output_path):
                with open(output_path, "r") as file:
                    plan = file.readlines()
                if len(plan) > 0:
                    return 0 # valid plan generated
                else:
                    return 1
            else:
                return 1
        except FileNotFoundError:
            return 1

def verify_predicates_lapkt_solver(
    output_path: str,
    pddl_domain_path: str,
    pddl_problem_path: str,
    solver: str,
    timeout: float = 5.0,
) -> bool:
    """The lapkt solver is a bit different than the fast downward solver, so we will
    verify the predicates in a different way."""
    pddl_dir = os.path.dirname(pddl_problem_path)

    if os.path.dirname(pddl_domain_path) != pddl_dir:
        new_temp_domain_path = os.path.join(pddl_dir, os.path.basename(pddl_domain_path))
        shutil.copy(pddl_domain_path, new_temp_domain_path)
        pddl_domain_path = new_temp_domain_path

    if (os.path.dirname(pddl_domain_path) != os.path.dirname(pddl_problem_path)
        or os.path.dirname(pddl_domain_path) != os.path.dirname(output_path)):
        raise Exception("All pddl files must be in the same directory due to how the files are copied into the docker container used for the solver.")

    # create the plan file that the planner will write to
    output_file = os.path.basename(output_path)

    # example command:
    # docker run --platform linux/amd64 --rm
    # -v ~/gpt-planner:/data lapkt/lapkt-public
    # timeout 30s ./ff --domain /data/alfworld_domain.pddl
    #                 --problem /data/out_problem.pddl
    #                 --output /data/plan.ipc

    arch = platform.machine().lower()
    if arch in ["x86_64", "amd64"]:
        docker_repo = "lapkt/lapkt-public"
    elif arch in ["arm64", "aarch64"]:
        docker_repo = "gautierdag/lapkt-arm"

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{os.path.abspath(pddl_dir)}:/data",
        f"{docker_repo}",
        "timeout",
        f"{timeout}s",
        f"./bfs_f",
        "--domain",
        f"/data/{os.path.basename(pddl_domain_path)}",
        "--problem",
        f"/data/{os.path.basename(pddl_problem_path)}",
        "--output",
        f"/data/{os.path.basename(output_file)}"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if "goal can be simplified to TRUE" in str(result.stdout):
        return True
    if "goal can be simplified to FALSE" in str(result.stdout):
        return False
    if "syntax error" in str(result.stderr):
        return False
    output_plan_file_path = os.path.join(pddl_dir, output_file)
    if os.path.exists(output_plan_file_path):
        # a plan was generated, sometimes, the generated plan will be empty
        # this means the goal state is satisfied
        with open(output_plan_file_path, "r") as file:
            plan = file.readlines()
        if len(plan) == 0:
            return True
        else:
            # a plan of non-zero length is needed to satisfy the preconditions
            return False
    else:
        # no valid plan could be found
        return False



def parse_pddl_to_function_kwargs(
    action_call: str, actions: List[ReactActionSchema]
) -> Tuple[Callable, Dict]:
    """Takes in a pddl action string from a generated plan and
    outputs the function and dictionary representation of the
    function args.

    Args:
        action (str): a pddl action call in the following format:
            (action_name arg_1 arg_2 arg_3)

    Returns:
        func (Callable): the function specified to call in the pddl
        args (Dict): the arguments to the function in dictionary format
    """
    # parse the pddl action name and arguments from the action call
    # provided in string format
    action_split = (
        action_call.replace("(", "").replace(")", "").replace("\n", "").split(" ")
    )
    pddl_action_name = action_split[0]
    pddl_arg_vals = action_split[1:]

    # get the corresponding action schema
    action_schema = [
        a for a in actions if a.pddl_action_name is not None and a.pddl_action_name.lower() == pddl_action_name
    ][0]
    python_action_func = action_schema.action_function
    domain_path = action_schema.pddl_domain_path
    pddl_action_arg_mappings = action_schema.pddl_action_arg_mappings
    if domain_path is None:
        raise ValueError(
            "Must initialize domain path in react action schema if using pddl planner."
        )
    if pddl_action_arg_mappings is None:
        raise ValueError(
            "Must initialize pddl_action_arg_mappings in react action schema if using pddl planner."
        )

    # match the pddl action arguments to the arguments in the true
    # python action function
    block_identifier = f"(:action {action_schema.pddl_action_name}"
    action_block = get_pddl_block_from_pddl_file(domain_path, block_identifier)
    pddl_arg_names = get_subblock_from_action_block_str(action_block, ":parameters")
    pddl_arg_names = (
        pddl_arg_names.replace(":parameters ", "").replace("(", "").replace(")", "")
    )

    pddl_arg_names = [param for param in pddl_arg_names.split(" ") if param[0] == "?"]

    python_function_kwargs = dict()
    for idx, p in enumerate(pddl_arg_names):
        m_list = [i for i in pddl_action_arg_mappings if i.pddl_arg == p]
        if len(m_list) != 1:
            raise ValueError(f"No pddl argument defined for {p}")
        mapping = m_list[0]
        if mapping.llm_arg is not None:
            python_function_kwargs[mapping.llm_arg] = pddl_arg_vals[idx]

    return python_action_func, python_function_kwargs


if __name__ == "__main__":
    pddl_file_path = "/Users/byrdgb1/Desktop/Projects/Concept_Agent/Code/concept-agent/outputs/default_experiment_name/2024-12-19/12-05-14/Test_Task_0002/Test_Task_0002_plans/Test_Task_0002_problem.pddl"

    check_if_goal_state_satisfied(
        pddl_file_path,
        "/Users/byrdgb1/Desktop/Projects/Concept_Agent/Code/concept-agent/src/utils/classical_planner/domain_files/ai2thor_domain.pddl",
    )
