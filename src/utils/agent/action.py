"""
A class for standardizing agent calls to actions.
"""

import warnings
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from colorama import Fore, Style

class AgentAction:
    """This class provides a standardized object for organizing
    information about an action called by an agent.
    """

    def __init__(
        self,
        action_type: str,
        func: Callable,
        agent_id: int,
        llm_planning_thought: str,
        llm_action_name: str,
        llm_action_input: Dict,
        pddl_action_name: Optional[str] = None,
        pddl_action_input: Optional[Dict] = None,
    ):
        self.action_type = action_type
        self.func = func
        self.agent_id = agent_id
        self.llm_planning_thought = llm_planning_thought
        self.llm_action_name = llm_action_name
        self.llm_action_input = llm_action_input
        self.pddl_action_name = pddl_action_name
        self.pddl_action_input = pddl_action_input

        # the timestamp will be set immediately after the action is
        # enacted
        self.observation: str | None = None
        self.return_flag: int | None = None
        self.timestamp: datetime | None = None

    async def call_action(self) -> Tuple[int, str]:
        self.failure_prefix = "Action Failed: "
        self.success_prefix = "Action Completed Successfully: "
        try:
            flag, observation = await self.func(
                **self.llm_action_input, agent_id=self.agent_id
            )
            if flag != 0:
                self.observation = Fore.RED + self.failure_prefix + observation + Style.RESET_ALL
            else:
                self.observation = Fore.GREEN + self.success_prefix + observation + Style.RESET_ALL
            self.return_flag = flag
            self.timestamp = datetime.now()
            assert isinstance(self.observation, str)
            return self.return_flag, self.observation
        except Exception as e:
            self.observation = Fore.RED + self.failure_prefix + str(e) + Style.RESET_ALL
            return 4, self.observation
        # flag, observation = await self.func(
        #     **self.llm_action_input, agent_id=self.agent_id
        # )
        if flag != 0:
            self.observation = self.failure_prefix + observation
        else:
            self.observation = self.success_prefix + observation
        self.return_flag = flag
        self.timestamp = datetime.now()
        assert isinstance(self.observation, str)
        return self.return_flag, self.observation

    def get_action_info_string(self):
        if self.observation is None:
            warnings.warn("Observation None as action has not yet been called.")
        return (
            f"\tThought: {self.llm_planning_thought}\n"
            f"\tAction: {self.llm_action_name}\n"
            f"\tAction Input: {self.llm_action_input}\n"
            f"\tObservation: {self.observation}\n\n"
        )

    def get_action_info_dict(self):
        return {
            "action_type": self.action_type,
            "agent_id": self.agent_id,
            "llm_planning_thought": self.llm_planning_thought,
            "llm_action_name": self.llm_action_name,
            "llm_action_input": self.llm_action_input,
            "pddl_action_name": self.pddl_action_name,
            "pddl_action_input": self.pddl_action_input,
            "observation": self.observation,
            "return_flag": self.return_flag,
            "timestep": str(self.timestamp),
        }


class TaskCompletionAction(AgentAction):

    def __init__(self, agent_id: int, observation=None, action_type="n/a"):
        super().__init__(
            action_type,
            self.print_task_complete,
            agent_id,
            "Task Complete!",
            "TaskComplete",
            dict(),
            "",
            dict(),
        )
        if observation is None:
            self.observation = "Task Complete!"
        else:
            self.observation = f"Task Complete! {observation}"
        self.return_flag = 0
        self.timestamp = datetime.now()

    def print_task_complete(self, **kwargs):
        print("Task Complete!")
