import random
from datetime import datetime
from typing import Dict, List, Set

from src.utils.agent.action import AgentAction
from src.utils.state.object_database import ObjectDatabase


class AgentState:
    """Maintains the state of an agent as well as various functions for
    interacting with and getting information about the state of the agent.

    Some functions will need to be called by the ReAct agent to query for
    detailed information, although some default information about the
    agent state will be provided at each tool calling iteration.
    """

    def __init__(self, agent_id: int):
        self.action_call_history = History()
        self.agent_id = agent_id
        self.agent_id_str = f"agent_{agent_id}"
        self.reset()

    def reset(self) -> None:
        self.action_call_history.reset()
        # the held object with always be None or an llm object id
        # specifying the object that the agent is holding
        self.held_obj: str | None = None
        # a list of objects that are visible to the agent visible objects
        self.visible_objects: Set = set()
        # the agent's location for planning will be determined by an object
        # that it is currently at
        self.cur_loc = ""

    def get_agent_state_default(self) -> Dict:
        """Returns a dictionary representing the state of the agent at a
        default level of detail. Intended to be provided to the agent as
        context at each tool calling iteration."""
        return {
            "held_object": self.held_obj,
            "visible_objects": self.visible_objects,
            "current_location": self.cur_loc,
        }

    def get_state_in_pddl_format(self, object_state: ObjectDatabase) -> List[str]:
        """Returns the agent state in a List of strings where each
        string is a predicate that represents the state of the agent.
        """
        pddl_state = list()

        # account for held objects
        if self.held_obj is not None:
            held_obj_ai2thor_id = object_state.ensure_llm_id(self.held_obj)
            pddl_state.append(f"(objectHeld agent_{self.agent_id} {held_obj_ai2thor_id})\n")
            pddl_state.append(f"(isHoldingObject agent_{self.agent_id})\n")

        # account for current location

        # if the current location is None, we will just choose a random
        # ai2thor object id as the starting location
        if self.cur_loc is None:
            cur_loc = random.choice(list(object_state.llm_ids))

        # the current location is an object id specified in LLM facing
        # coordinates, so we will convert this to ai2thor object id
        else:
            cur_loc = object_state.ensure_llm_id(self.cur_loc)
        pddl_state.append(f"(atLocation agent_{self.agent_id} {cur_loc})\n")

        return pddl_state


class History:
    """Maintains the history of an agent's action calls."""

    def __init__(self):
        self.history: List[AgentAction] = list()

    def reset(self) -> None:
        self.history: List[AgentAction] = list()

    def add_action(self, action: AgentAction) -> None:
        """Add an action call to the action call history of an agent.

        Args:
            action (AgentAction): the action to add to the history

        Returns:
            None
        """
        self.history.append(action)

    def get_history_llm_facing(self) -> str:
        """Returns the llm facing history in string format to pass
        to an LLM React agent.

        Args:
            None

        Returns:
            llm_facing_history (str): the action call history of the agent in str format
                to be passed to an llm react agent
        """
        llm_facing_history = list()
        for a in self.history:
            thought_line = f"Thought: {a.llm_planning_thought}"
            action_line = f"Action: {a.llm_action_name}"
            action_input = f"Action Input: {a.llm_action_input}"
            observation = f"Observation: {a.observation}"
            temp = [thought_line, action_line, action_input, observation]
            llm_facing_history.append(("\n").join(temp))

        llm_facing_history = ("\n\n").join(llm_facing_history)
        return llm_facing_history
