"""
Define all action call functions for the Alfworld environment.
"""

import re
from typing import Tuple

from src.utils.environment.alfworld.env import AlfworldEnv
from src.utils.llms.load import get_llm


# ONLY ACCESSIBLE BY LLM
async def search_for_object(target_object_type: str, agent_id: int) -> Tuple[int, str]:
    """A tool for searching for an object in the environment."""

    # in the event he LLM ID was passed on accident instead of the target ID,
    # we will process this below
    if "_" in target_object_type:
        object_type = target_object_type.split("_")[0]
    target_object_type = target_object_type.replace(" ", "")

    obj_db = AlfworldEnv.obj_db
    # no objects with the target type have been discovered yet, so we will search

    # get a list of all receptacles
    receptacle_types = set(
        [
            v["objectType"]
            for k, v in obj_db.obj_metadatas.items()
            if ("isReceptacle" in v and v["isReceptacle"]) and ("searched" not in v or not v["searched"])
        ]
    )

    previously_disc_objs_w_target_type = [x for x in obj_db.llm_ids if target_object_type.lower() in x.lower()]

    if len(receptacle_types) == 0:
        # if objects of the specified type have previously been discovered, 
        # we want to note this
        if len(previously_disc_objs_w_target_type) > 0:
            return 0, f"All objects have been discovered. Previously discovered objects of type {target_object_type} include {', '.join(previously_disc_objs_w_target_type)}."
        else:
            return 1, f"All objects have been discovered. No objects of type {target_object_type} were discovered."

    # prompt the LLM for most likely receptacles to search inside
    llm = AlfworldEnv.agents[agent_id].llm

    prompt = """Given a target object, you are tasked with ordering a list of possible receptacles
    from most likely to least likely to contain the target object. Strictly follow the following format:


    Target Object: the target object
    Possible Receptacle: a list of possible receptacles
    Likely Receptacles: [most likely receptacle, ... , least likely receptacle]
    Done!

    Begin!

    Target Object: {target_object}
    Possible Receptacle: {receptacle_types}
    """
    input = {"target_object": target_object_type, "receptacle_types": receptacle_types}
    formatted = prompt.format(**input)
    max_retries = 3
    llm_success = False
    ordered_receptacles = list()
    for _ in range(max_retries):
        llm_output = llm.invoke(prompt=formatted)

        # parse the output
        likely_receptacles_pattern = r"Likely Receptacles:\s*\[(.*)\]"
        likely_match = re.search(likely_receptacles_pattern, llm_output.content)
        if likely_match:
            likely_receptacles = likely_match.group(1).strip()
            ordered_receptacles = [
                item.strip().replace("'", "").replace("\"", "") for item in likely_receptacles.split(",")
            ]
        else:
            continue  # retry

        # perform some checks on the parsed data
        # remove any erroneous receptacles proposed by the language model
        filtered_ordered_receptacles = list()
        for receptacle in ordered_receptacles:
            if receptacle in receptacle_types:
                filtered_ordered_receptacles.append(receptacle)
        ordered_receptacles = filtered_ordered_receptacles

        # if the LLM does not include all possible search locations in its
        # proposal, add the search locations left out to the end of the ordered
        # receptacle list
        mismatched_receptacle_types = set(ordered_receptacles) ^ receptacle_types
        if len(mismatched_receptacle_types) != 0:
            ordered_receptacles.extend(list(mismatched_receptacle_types))
        
        if len(set(ordered_receptacles) ^ receptacle_types) != 0:
            raise Exception("Proposed ordered search locations should equal possible search locations.")
        llm_success = True
        break

    if not llm_success:
        raise Exception("The LLM could not re-order receptacles in the correct format.")

    # search through all receptacles of a certain type based on the likelihood of
    # finding the target object contained in that receptacle

    # below is a dict that will map receptacles with objects that were discovered
    # in or on that receptacle
    discovered_objs = dict()
    target_object_found = False
    for receptacle_type in ordered_receptacles:
        receptacle_ids = [
            v["objectId"]  # environment ID
            for k, v in obj_db.obj_metadatas.items()
            if (v["objectType"] == receptacle_type and ("searched" not in v or not v["searched"]))
        ]
        for receptacle_id in receptacle_ids:
            # move to the target receptacle
            move_to_return_flag, obs = await go_to_receptacle(receptacle_id, agent_id)

            if move_to_return_flag != 0:  # the move to command did not work
                raise Exception(
                    "Need to determine how to address this in the search action function."
                )

            # if the receptacle is closed, you must open it to search it
            if "closed" in obs.lower():
                open_return_flag, obs = await open_receptacle(receptacle_id, agent_id)

                if open_return_flag != 0:  # the open command did not work
                    raise Exception(
                        "Need to determine how to address this in the search action function."
                    )

            # get the list of objects discovered
            processed_obs_dict = AlfworldEnv.process_obs(obs)

            if len(processed_obs_dict.keys()) > 1:
                raise Exception(
                    "This was unanticipated and how to handle needs to be decided."
                )
            
            if len(processed_obs_dict.keys()) == 0:
                continue

            # add discovered objects information to the dictionary
            discovered_objs = discovered_objs | processed_obs_dict

            # if the target object has been found, you can break
            newly_discovered_objs = list(processed_obs_dict.values())[0]
            if target_object_type in [
                v.split("-")[0] for v in newly_discovered_objs
            ]:
                target_object_found = True
                break

        if target_object_found:
            break

    # remove all keys in the discovered objects dictionary for receptacles
    # that did not contain any new objects
    discovered_objs = {k: v for k, v in discovered_objs.items() if len(v) > 0}

    # prepare the return observation that announces all discovered objects
    if not target_object_found:
        if len(previously_disc_objs_w_target_type) > 0:
            return 0, f"No new objects of type {target_object_type} could be found, but previously discovered objects of type {target_object_type} include {', '.join(previously_disc_objs_w_target_type)}."
        else:
            return 1, f"No objects of type {target_object_type} could be found!"

    obs = f"{target_object_type} found! Here is a list of newly discovered objects: "
    for receptacle, new_objs in discovered_objs.items():
        new_objs_llm_ids = [AlfworldEnv.obj_db.ensure_llm_id(x) for x in new_objs]
        obs += f"Found {', '.join(new_objs_llm_ids)} in or on {AlfworldEnv.obj_db.ensure_llm_id(receptacle)}. "

    return 0, obs


async def examine_object_in_light(
    held_object_id: str, light_source_object_id: str, agent_id: int
) -> Tuple[int, str]:
    held_object_env_id = AlfworldEnv.obj_db.ensure_env_id(held_object_id)
    light_source_object_env_id = AlfworldEnv.obj_db.ensure_env_id(
        light_source_object_id
    )
    alfworld_action = (
        f"examineinlight {held_object_env_id} {light_source_object_env_id}"
    )
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not examine {held_object_id}."
    return 0, observation


async def go_to_receptacle(receptacle_id: str, agent_id: int) -> Tuple[int, str]:
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"go to {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not go to {receptacle_id}."
    return 0, observation


async def open_receptacle(receptacle_id: str, agent_id: int) -> Tuple[int, str]:
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"open {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not open {receptacle_id}."
    return 0, observation


async def close_receptacle(receptacle_id: str, agent_id: int) -> Tuple[int, str]:
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"close {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not close {receptacle_id}."
    return 0, observation


async def pickup_object_from_receptacle(
    object_id: str, receptacle_id: str, agent_id: int
) -> Tuple[int, str]:
    object_env_id = AlfworldEnv.obj_db.ensure_env_id(object_id)
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"take {object_env_id} from {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not take {object_id} from {receptacle_id}."
    return 0, observation


async def put_object(
    object_id: str, receptacle_id: str, agent_id: int
) -> Tuple[int, str]:
    object_env_id = AlfworldEnv.obj_db.ensure_env_id(object_id)
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"move {object_env_id} to {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not put {object_id} in or on {receptacle_id}."
    return 0, observation


async def clean_object(
    object_id: str, receptacle_id: str, agent_id: int
) -> Tuple[int, str]:
    object_env_id = AlfworldEnv.obj_db.ensure_env_id(object_id)
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"clean {object_env_id} with {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not clean {object_id} with {receptacle_id}."
    return 0, observation


async def heat_object(
    object_id: str, receptacle_id: str, agent_id: int
) -> Tuple[int, str]:
    object_env_id = AlfworldEnv.obj_db.ensure_env_id(object_id)
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"heat {object_env_id} with {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not heat {object_id} with {receptacle_id}."
    return 0, observation


async def cool_object(
    object_id: str, receptacle_id: str, agent_id: int
) -> Tuple[int, str]:
    object_env_id = AlfworldEnv.obj_db.ensure_env_id(object_id)
    receptacle_env_id = AlfworldEnv.obj_db.ensure_env_id(receptacle_id)
    alfworld_action = f"cool {object_env_id} with {receptacle_env_id}"
    observation, reward, done, info = AlfworldEnv.step(
        agent_id=agent_id, alfworld_action=[alfworld_action]
    )
    if "Nothing happens" in observation:
        return 3, f"Could not cool {object_id} with {receptacle_id}."
    return 0, observation
