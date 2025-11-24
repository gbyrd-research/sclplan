"""
Defines the class for a Classical Planner Agent.
"""

import os
from typing import Dict, List, Optional, Tuple

from colorama import Fore, Style

from src.tasks.base import TaskMetadata
from src.utils.agent.action import AgentAction, TaskCompletionAction
from src.utils.classical_planner import pddl
from src.utils.classical_planner.pddl import ReactActionSchema
from src.utils.environment.base import Environment
from src.utils.misc import logging_utils
from src.utils.state.agent_state import AgentState

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)
PLANNER = os.path.join(
    CUR_DIR, "..", "classical_planner", "fast-downward-22.12/fast-downward.py"
)


class ClassicalAgent:
    """A basic classical planner agent that takes in a state representation
    of the environment and a goal state and attempts to generate a plan
    trajectory that will reach the goal state."""

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        pddl_domain_file_path: str,
        solver: str,
        actions: List[ReactActionSchema],
        state: AgentState,
        logger
    ):
        self.agent_id = agent_id
        self.env = env
        self.domain = pddl_domain_file_path
        self.solver = solver
        self.actions = actions
        self.state = state
        self.logger = logger

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
            obj_metadata (List[Dict]): a list of the metadata of all discovered
                objects in the scene

        Returns:
            agent_id (int): the id of this agent class object
            final_answer (str): the final response from the LLM agent
            agent_state_in_dict_format (Dict): the state of the agent
                in dictionary format
        """
        self._set_logging_dir(task_metadata, global_task_metadata)
        if task_metadata.pddl_goal_state is None:
            raise ValueError(
                "If using the classical planning agent, must define valid pddl goal state in the task metadata object."
            )
        goal_block = task_metadata.pddl_goal_state
        task_name = task_metadata.task_name
        plan_save_dir = os.path.join(self.logging_dir, f"{task_name}_plans")
        pddl_problem_save_path = os.path.join(
            plan_save_dir, f"{task_name}_problem.pddl"
        )
        logging_utils.ensure_dir_exists(plan_save_dir)
        plan_save_path = os.path.join(plan_save_dir, f"{task_name}_plan.txt")
        pddl.create_pddl_problem_file(
            goal_block, pddl_problem_save_path, self.domain, self.env
        )
        result = pddl.create_plan(plan_save_path, self.domain, pddl_problem_save_path, self.solver)
        if result != 0:
            print(Fore.RED + f"{task_metadata.task_name}: Could not generate plan!")
            return self._format_perform_task_return(result)
        print(Fore.GREEN + f"Valid plan generated and saved to {plan_save_path}!")
        print(Style.RESET_ALL)
        result, obs = await self.enact_plan(plan_save_path, self.solver)
        print(Fore.GREEN + f"Plan enacted successfully!" + Style.RESET_ALL)
        return self._format_perform_task_return(result)

    async def enact_plan(self, plan_fp: str, solver: str, action_type: str = "classical") -> Tuple[int, str]:
        """Given a plan.txt file, attempt to parse the plan from the file
        and enact the plan in the ai2thor world.

        Args:
            plan_fp (str): the file path to the generated plan.txt file

        Returns:
            flag (int): specifies the result of enacting the plan
                0: plan was successfully enacted
                1: plan could not be enacted successfully
        """

        with open(plan_fp, "r") as file:
            plan = file.readlines()
            # remove the plan cost information that is found on the last line of the generated plan
            if solver == "fast_downward":
                plan = plan[:-1]
        
        # there is some weird thing where the planner will output a plan with
        # (REACH-GOAL)\n as a pddl action... here we address this
        processed_plan = list()
        for action in plan:
            if "REACH-GOAL" in action:
                continue
            processed_plan.append(action)
        plan = processed_plan

        for lowercase_pddl_action_call in plan:

            lowercase_pddl_action_call = lowercase_pddl_action_call.replace("(", "").lower()
            lowercase_pddl_action = lowercase_pddl_action_call.split(" ")[0]

            # get the matching action schema
            action_schema = None
            for a in self.actions:
                if a.pddl_action_name is None:
                    # raise ValueError(
                    #     "If using classical planner, you must define all pddl values in the action schemas."
                    # )
                    continue
                if a.pddl_action_name.lower() == lowercase_pddl_action:
                    action_schema = a
                    break
            if action_schema is None:
                raise ValueError(
                    f"No matching pddl action found for {lowercase_pddl_action} in action schemas."
                )

            llm_action_name = action_schema.llm_action_name

            # convert the pddl action input to LLM action input
            _, llm_action_input = pddl.parse_pddl_to_function_kwargs(
                lowercase_pddl_action_call, self.actions
            )

            # ensure tool functions recieve llm object ids as input
            obj_db = self.env.obj_db
            for param, val in llm_action_input.items():
                if isinstance(val, str):
                    if val.lower() in [
                        x.lower() for x in obj_db.llm_ids
                    ] or val.lower() in [x.lower() for x in obj_db.env_ids]:
                        llm_action_input[param] = self.env.obj_db.ensure_llm_id(val)

            # get the pddl action input from the python function input
            pddl_action_input = pddl.get_pddl_action_input(
                llm_action_name, llm_action_input, self.actions, self.state, self.env
            )

            # build the AgentAction
            agent_action = AgentAction(
                action_type,
                action_schema.action_function,
                self.agent_id,
                llm_planning_thought="",  # a classical planner does not reason, therefore no thought provided
                llm_action_name=llm_action_name,
                llm_action_input=llm_action_input,
                pddl_action_name=action_schema.pddl_action_name,
                pddl_action_input=pddl_action_input,
            )

            action_flag, obs = await agent_action.call_action()

            self.logger.info(
                "Classical Agent Predicted Action:\n" + agent_action.get_action_info_string()
            )

            self.state.action_call_history.add_action(agent_action)

            if action_flag != 0:
                return action_flag, obs

        task_complete_action = TaskCompletionAction(self.agent_id)

        # self.state.action_call_history.add_action(task_complete_action)
        assert task_complete_action.observation is not None
        return 0, task_complete_action.observation

    def _format_perform_task_return(self, result: int) -> Tuple[int, str, Dict]:
        """Formats the return value for the self.perform_task function based
        on the provided result from the pddl planning and enacting steps.

        Args:
            result (int): the resulting return flag from either the self._create_plan()
                method or the self._enact_plan() method

        Returns:
            agent_id (int): the id of this agent class object
            final_answer (str): the final response from the LLM agent
            agent_state_in_dict_format (Dict): the state of the agent
                in dictionary format
        """
        # TODO: Need to determine what natural language observation to return
        # based on the return flags from the pddl generation/enact_plan processes
        final_answer = "" if result != 0 else "Plan enacted succesfully."
        agent_state_in_dict_format = self.state.get_agent_state_default()
        return result, final_answer, agent_state_in_dict_format

    def _set_logging_dir(
        self, task_metadata: TaskMetadata, global_task_metadata: TaskMetadata | None
    ) -> None:
        hydra_output_dir = logging_utils.get_hydra_run_dir()
        if global_task_metadata is None:
            self.logging_dir = os.path.join(hydra_output_dir, task_metadata.task_name)
        else:
            self.logging_dir = os.path.join(
                hydra_output_dir, global_task_metadata.task_name
            )
