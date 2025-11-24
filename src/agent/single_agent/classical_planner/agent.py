"""
Defines a Classical Planner agent for use in Single Agent tasks in AI2Thor.
"""

import os
from typing import Dict, Optional, Tuple

from src.agent.single_agent.base import BaseAgent
from src.tasks.base import TaskMetadata
from src.utils.agent.action import TaskCompletionAction
from src.utils.agent.classical_planner import ClassicalAgent
from src.utils.base_utils import get_obj_from_str
from src.utils.classical_planner import pddl
from src.utils.misc import logging_utils
from src.utils.state.agent_state import AgentState


class ClassicalPlannerAgent(BaseAgent):

    def __init__(
        self,
        env_target: str,
        react_action_schemas_target: str,
        pddl_domain_path: str,
        solver: str,
        agent_id: int,
    ):
        """An implementation of a Classical Planner for the AI2Thor
        environment.

        Args:
            agent_id (int): the id of the agent
        """
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.state = AgentState(agent_id)
        self.env = get_obj_from_str(env_target)
        self.actions = get_obj_from_str(react_action_schemas_target)
        self.pddl_domain_path = pddl_domain_path
        self.solver = solver
        self.cp = ClassicalAgent(
            agent_id=self.agent_id,
            env=self.env(),
            pddl_domain_file_path=self.pddl_domain_path,
            solver=self.solver,
            actions=self.actions,
            state=self.state,
        )

    def get_total_token_count(self) -> int:
        """The classical planner does not use an LLM so the token count is 0."""
        return 0

    def get_action_sequence_length(self) -> int:
        return len(self.state.action_call_history.history)

    async def perform_task(
        self,
        task_metadata: TaskMetadata,
        global_task_metadata: Optional[TaskMetadata] = None,
    ) -> Tuple[int, str, Dict]:
        """Perform a given task.

        Args:
            task_metadata (TaskMetadata): information about the task
                the agent is expected to perform
            global_task_metadata (Optional[TaskMetadata]): this optional
                argument is for cases when the agent is part of a
                multi-agent team. In that case, the provided task is
                likely a delegated subtask, which is different than the
                global, high level task was was initially provided to the
                system. The global task information is important for some
                functions, therefore it is included if necessary.

        Returns:
            agent_id (int): the id of this agent class object
            final_answer (str): the final response from the LLM agent
            agent_state_in_dict_format (Dict): the state of the agent
                in dictionary format
        """
        self._set_logging_dir(task_metadata, global_task_metadata)

        # boilerplate
        self.task_metadata = task_metadata
        self.global_task_metadata = global_task_metadata
        self.goal_block = self.task_metadata.pddl_goal_state
        self.task_name = task_metadata.task_name
        self.plan_save_dir = os.path.join(self.logging_dir, f"{self.task_name}_plans")
        self.pddl_problem_file_path = os.path.join(
            self.plan_save_dir, f"{self.task_name}_problem.pddl"
        )
        self.plan_save_path = os.path.join(
            self.plan_save_dir, f"{self.task_name}_plan.txt"
        )
        logging_utils.ensure_dir_exists(self.plan_save_dir)

        # use the goal state and environment state to generate a
        # pddl problem file for use with the classical planner
        pddl.create_pddl_problem_file(
            self.goal_block,
            self.pddl_problem_file_path,
            self.pddl_domain_path,
            self.env(),
        )

        # attempt to create a plan
        cp_result = pddl.create_plan(
            self.plan_save_path, 
            self.pddl_domain_path, 
            self.pddl_problem_file_path, 
            self.solver
        )

        if cp_result == 0:  # valid plan found

            self.logger.info(
                "Valid plan found! Attempting to enact plan with classical planner."
            )

            # attempt to enact the plan
            current_plan_len = len(self.state.action_call_history.history)
            return_flag, enact_plan_obs = await self.cp.enact_plan(self.plan_save_path, self.solver)

            if return_flag == 0:  # plan enacted successfully
                agent_action = TaskCompletionAction(self.agent_id)
                self.state.action_call_history.add_action(agent_action)

                self.logger.info("Plan enacted successfully with classical planner!")

                for a in self.state.action_call_history.history[current_plan_len:]:
                    self.logger.info(
                        "Classical Planner Predicted Action:\n"
                        + agent_action.get_action_info_string()
                    )

                assert isinstance(agent_action.observation, str)
                return (
                    self.agent_id,
                    agent_action.observation,
                    self.state.get_agent_state_default(),
                )

            else:  # error occurred while enacting plan

                self.logger.info(
                    f"Unable to enact plan. Recieved error: {enact_plan_obs}!"
                )

                for a in self.state.action_call_history.history[current_plan_len:]:
                    self.logger.info(
                        "Classical Planner Predicted Action:\n"
                        + a.get_action_info_string()
                    )

                return (
                    self.agent_id,
                    f"Task Failed: The following error occurred while attempting to enact the plan: {enact_plan_obs}.",
                    self.state.get_agent_state_default(),
                )

        else:  # a valid plan could not be found by the classical planner

            self.logger.info("Valid plan could not be found.")

            return (
                self.agent_id,
                "Task Failed: Valid plan could not be found by classical planner.",
                self.state.get_agent_state_default(),
            )
