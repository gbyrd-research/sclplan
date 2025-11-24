"""
Defines the (C)lassically (A)ugmented (L)LM (A)gent or CALA.
"""

import json
import time
import logging
import os
import signal
import re
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.environment.alfworld.env import AlfworldEnv
from src.agent.single_agent.base import BaseAgent
from src.agent.single_agent.CALA.prompt import (GOAL_STATE_TEMPLATE_AI2THOR,
                                                GOAL_STATE_TEMPLATE_ALFWORLD,
                                                GENERATE_GOAL_PROMPT_LLMDP)
from src.tasks.base import TaskMetadata
from src.utils.agent.action import AgentAction, TaskCompletionAction
from src.utils.agent.classical_planner import ClassicalAgent
from src.utils.agent.react_w_pddl import ReactAgentWithPDDL
from src.utils.base_utils import get_obj_from_str
from src.utils.classical_planner import pddl
from src.utils.classical_planner.pddl import get_pddl_block_from_pddl_file
from src.utils.environment.base import Environment
from src.utils.llms.load import get_llm
from src.utils.misc import formatting, logging_utils
from src.utils.misc.create_media import create_task_media
from src.utils.misc.logging_utils import get_hydra_run_dir
from src.utils.state.agent_state import AgentState
from src.utils.state.goal_state_functions import (get_goal_state_helper_descs,
                                                  valid_pddl_predicates)
from src.utils.classical_planner.pddl import (
    get_pddl_action_input,
    get_pddl_action_name,
)

from colorama import Fore, Style

COPY_LLMDP_ALFWORLD_GOAL_PROMPT = 1

class CALA(BaseAgent):
    """Implementation of the Classically Augmented LLM Agent
    which is an LLM agent that leverages classical planning
    to improve robustness, planning speed, and significantly
    reduces token count."""

    def __init__(
        self,
        llm: str,
        env_target: str,
        react_action_schemas_target: str,
        pddl_domain_path: str,
        agent_id: int,
        solver: str,
        max_planning_steps: int = 35,
        results_dir: str = None,
        use_classical_planner: bool = True,
        use_precondition_verification: bool = True,
        llm_kwargs: dict = None,
        human_control_for_debugging: bool = False
    ):
        self.env: Environment = get_obj_from_str(env_target)()
        super().__init__(agent_id, self.env, results_dir)

        # randomize seed if None is given
        if llm_kwargs.seed is None:
            llm_kwargs.seed = random.randint(0, 10000)

        self.agent_id = agent_id
        self.actions = get_obj_from_str(react_action_schemas_target)
        self.pddl_domain_path = pddl_domain_path
        self.solver = solver
        self.state = AgentState(agent_id)
        self.max_planning_steps = max_planning_steps
        self.domain_predicate_info = self.parse_domain_predicate_info(pddl_domain_path)
        self.use_classical_planner = use_classical_planner
        self.use_precondition_verification = use_precondition_verification
        self.human_control_for_debugging = human_control_for_debugging
        # TODO: this is a really hacky way to do this.. need to refactor code such that
        # all LLMs are the same LLM per agent. currently, there are many different
        # LLMs working for the goal planner, react agent, and general CALA funtionality
        # this is confusing and hard to synchronize
        self.llm = get_llm(llm, **llm_kwargs)
        domain_file = [str(part) for part in Path(pddl_domain_path).parts][-1]
        domain_file_par_dir = "/".join(pddl_domain_path.split("/")[:-1])
        try:
            example_target = os.path.join(
                domain_file_par_dir,
                f"{domain_file.split('.')[0]}_pddl_generation_examples.EXAMPLES",
            ).replace("/", ".")
            self.goal_generation_examples = get_obj_from_str(example_target)
        except:
            raise Exception(
                "No goal generation example file for this pddl domain found!"
            )

        # define classical planner component and goal state LLM
        self.classical_planner = None
        self.goal_state_llm = None
        if self.use_classical_planner:
            # define Goal State generating LLM
            self.goal_state_llm = GoalStateLLM(llm, self.env, self.logger, max_runs=5)

        if self.use_classical_planner or self.use_precondition_verification:
            self.cp = ClassicalAgent(
                agent_id=agent_id,
                env=self.env,
                pddl_domain_file_path=self.pddl_domain_path,
                solver=self.solver,
                actions=self.actions,
                state=self.state,
                logger=self.logger,
            )

        # define react component
        self.react_agent = ReactAgentWithPDDL(
            agent_id=agent_id,
            env=self.env,
            llm=llm,
            actions=self.actions,
            state=self.state,
            logger=self.logger,
            llm_kwargs=llm_kwargs
        )

    def get_total_token_count(self) -> int:
        total_token_count = 0
        for llm_response in self.react_agent.model_response_list:
            total_token_count += llm_response.token_count
        if self.use_classical_planner:
            total_token_count += self.goal_state_llm.token_count
        for llm_response in self.llm.response_history:
            total_token_count += llm_response.token_count
        return total_token_count

    def get_action_sequence_length(self) -> int:
        # we do not want to count a "Task Completion Action"
        if self.state.action_call_history.history[-1].llm_action_name == "TaskComplete":
            return len(self.state.action_call_history.history[:-1])
        return len(self.state.action_call_history.history)
        
    def get_env_steps(self) -> int:
        return len(self.env.env_steps)
    
    def log_env_steps(self) -> None:
        env_steps_dict = {i: x for i, x in enumerate(self.env.env_steps)}
        with open(os.path.join(self.logging_dir, "env_steps.json"), "w") as file:
            json.dump(env_steps_dict, file, indent=3)


    def reset(self) -> None:
        """Reset the Agent for the next task."""

        # clear the React Agent's response list
        self.state.action_call_history.reset()
        self.react_agent.reset()
        if self.use_classical_planner:
            self.goal_state_llm.reset()
        self.llm.reset()
        self.planning_loop_idx = 0

        # reset quantitative metrics
        precond_reset_val = 0 if self.use_precondition_verification else "n/a"
        self.react_proposed_w_unsatisfied_preconditions = precond_reset_val
        self.react_proposed_w_satisfied_preconditions = precond_reset_val
        self.precondition_fails_to_generate_plan = precond_reset_val
        self.precondition_valid_plan_generated = precond_reset_val
        self.precondition_plan_enacted_successfully = precond_reset_val
        self.precondition_plan_not_enacted_successfully = precond_reset_val 

        global_planner_reset_val = 0 if self.use_classical_planner else "n/a"
        self.global_planner_fails_to_generate_plan = global_planner_reset_val
        self.global_planner_generates_plan = global_planner_reset_val
        self.global_planner_plan_not_enacted_successfully = global_planner_reset_val
        self.global_planner_plan_enacted_successfully = global_planner_reset_val
        self.generated_goal_state_correct = global_planner_reset_val

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

        self.reset()

        # check to see if the results directory already exists. skip tasks that
        # have already been completed to avoid running the planner on a task more
        # than once during a specific experiment
        task_result_save_path = os.path.join(
            self.results_dir, task_metadata.task_name, "results.json"
        )
        if self.results_dir is not None and os.path.exists(task_result_save_path):
            return (
                self.agent_id,
                "Task result already found. Skipping",
                self.state.get_agent_state_default(),
            )

        self._set_logging_dir(task_metadata, global_task_metadata, self.results_dir)
        if "ai2thor" in self.env.__class__.__name__.lower():
            self.env.set_logging_dir(self.logging_dir)

        # boilerplate
        self.task_metadata = task_metadata
        self.global_task_metadata = global_task_metadata
        self.task_name = task_metadata.task_name

        self.pddl_save_dir = os.path.join(self.logging_dir, f"{self.task_name}_plans")
        self.pddl_problem_file_path = os.path.join(
            self.pddl_save_dir, f"{self.task_name}_problem.pddl"
        )
        self.plan_save_path = os.path.join(
            self.pddl_save_dir, f"{self.task_name}_plan.txt"
        )
        logging_utils.ensure_dir_exists(self.pddl_save_dir)

        if self.use_classical_planner:
            # if necessary, use llm to create goal state from the natural language task
            self.logger.info("Generating goal state for classical planner...")
            self.goal_block = self.goal_state_llm.get_pddl_goal_block(
                task_metadata, self.domain_predicate_info, self.goal_generation_examples
            )

            # now, use this goal state as well as the environmental state
            # representation to generate a pddl problem file for use with
            # the classical planner
            pddl.create_pddl_problem_file(
                self.goal_block,
                self.pddl_problem_file_path,
                self.pddl_domain_path,
                self.env,
            )

        # run planning loop
        final_obs = await self._planning_loop()

        return self.agent_id, final_obs, self.state.get_agent_state_default()

    async def _planning_loop(self) -> str:
        """The planning loop for the CALA agent is as follows:

        1. Attempt to generate a plan to reach the goal state with
            classical planning component.
        2. If no valid plan found, predict next action with ReAct
            agent
        3. Check the preconditions of the ReAct prediction with the
            corresponding action definition in the classical planner
            domain file. If preconditions are not satisfied, try to
            reach the goal state with the classical planner. If no
            action can be generated, return an observation to the
            ReAct agent specifying which precondition was not satisfied.
        4. Enact the action in the environment.

        Args:
            None

        Returns:
            final_observation (str): the final observation from the planning
        """

        task = self.task_metadata.task_natural_language

        # start_time = time.time()
        # # Code to be timed
        # time.sleep(2)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time} seconds")

        for self.planning_loop_idx in range(self.max_planning_steps):
            ####################################################################
            # PART 1 FROM DOCSTRING

            # create a temporary domain with updated static types for each class
            # of object

            updated_domain_path = self._create_updated_domain_file()

            if self.use_classical_planner:
                self.logger.info("Attempting to solve with classical planner...")
                # recreate the problem file with an updated state
                pddl.create_pddl_problem_file(
                    self.goal_block,
                    self.pddl_problem_file_path,
                    self.pddl_domain_path,
                    self.env,
                )
                cp_result = pddl.create_plan(
                    self.plan_save_path,
                    updated_domain_path,
                    self.pddl_problem_file_path,
                    self.solver,
                )
                if cp_result == 0:  # valid global plan found
                    self.global_planner_generates_plan += 1
                    self.logger.info(
                        "Valid plan found! Attempting to enact plan with classical planner."
                    )
                    # attempt to enact the plan
                    current_plan_len = len(self.state.action_call_history.history)
                    return_flag, obs = await self.cp.enact_plan(
                        self.plan_save_path, self.solver
                    )
                    if return_flag == 0:  # plan enacted successfully
                        self.global_planner_plan_enacted_successfully += 1
                        task_completion_agent_action = TaskCompletionAction(self.agent_id)
                        self.state.action_call_history.add_action(
                            task_completion_agent_action
                        )
                        self.logger.info(
                            "Plan enacted successfully with classical planner!"
                        )
                        for a in self.state.action_call_history.history[current_plan_len:]:
                            self.logger.info(
                                "Classical Planner Predicted Action:\n"
                                + a.get_action_info_string()
                            )
                        assert isinstance(task_completion_agent_action.observation, str)
                        return task_completion_agent_action.observation

                    else:  # valid plan was found, but there was an error when trying to enact the plan
                        self.global_planner_plan_not_enacted_successfully += 1
                        self.logger.info("Valid plan found, but there was an error while enacting that plan. Continuing with ReAct...")

                else:  # valid global plan not found
                    self.global_planner_fails_to_generate_plan += 1
                    self.logger.info("No valid plan found. Continuing with ReAct...")

            ####################################################################
            # PART 2 FROM DOCSTRING

            # if debugging, we will ask for a human input here
            if self.human_control_for_debugging:
                agent_action = self._get_ai2thor_action_from_human_input()

            else:
                # call the ReAct agent

                # there is some weird issue with using the llama family of models
                # where they just hang perpetually. below, I employ a timeout to fix this.
                action_recieved = False
                consecutive_timeouts = 0
                num_consecutive_timeouts_allowed = 3
                while not action_recieved and consecutive_timeouts < num_consecutive_timeouts_allowed:
                    signal.signal(signal.SIGALRM, lambda x: print("Action Call Timeout..."))
                    signal.alarm(20)
                    try:
                        agent_action = self.react_agent.get_next_action(task)
                        action_recieved = True
                    except:
                        consecutive_timeouts += 1
                if consecutive_timeouts >= num_consecutive_timeouts_allowed:
                    raise Exception("Max number of consecutive timeouts reached...")
                # if task completion, we break and finish
                if isinstance(agent_action, TaskCompletionAction):
                    self.state.action_call_history.add_action(agent_action)
                    assert isinstance(agent_action.observation, str)
                    return agent_action.observation

            ####################################################################
            # PART 3 - Check Preconditions (FROM DOCSTRING)

            if self.use_precondition_verification:
                preconds_verified = await self.verify_preconditions(agent_action, updated_domain_path)

            ####################################################################
            # PART 4 FROM DOCSTRING

            # attempt to enact the action predicted by the react planner
            action_return_flag, obs = await agent_action.call_action()

            if self.human_control_for_debugging:
                self.logger.info(
                    "Human User Chosen Action:\n" + agent_action.get_action_info_string()
                )
            else:
                self.logger.info(
                    "ReAct Predicted Action:\n" + agent_action.get_action_info_string()
                )
            self.state.action_call_history.add_action(agent_action)

        return "Agent reached max planning iterations."
    
    async def verify_preconditions(
            self, 
            agent_action: AgentAction,
            updated_domain_path: str
        ) -> bool:
        """Checks if the preconditions for a given ReAct-Proposed action are met.
        Attempts to satisfy those preconditions through formal planning if they
        are not initially met."""

        # print(agent_action.get_action_info_string())

        # some actions are LLM-facing only, meaning they cannot be or are not
        # defined in PDDL. In this case, precondition checking is unnecessary.
        if agent_action.pddl_action_name is None:
            self.react_proposed_w_satisfied_preconditions += 1
            self.logger.info("All preconditions satisfied...")
            return True
        
        self.logger.info(
            "Attempting to formally verify preconditions of ReAct predicted action..."
        )
        precond_verification_dir = os.path.join(self.logging_dir, "precond_ver")
        logging_utils.ensure_dir_exists(precond_verification_dir)

        # build a pddl problem file where the goal state is a state in
        # which the preconditions for the target action are met
        self.precondition_pddl_problem_path = os.path.join(
            precond_verification_dir, "precond_problem.pddl"
        )
        self.precondition_pddl_domain_path = os.path.join(
            os.path.dirname(self.precondition_pddl_problem_path),
            os.path.basename(updated_domain_path),
        )
        shutil.copy(updated_domain_path, self.precondition_pddl_domain_path)
        self.precondition_plan_path = os.path.join(
            precond_verification_dir, "precon_plan.txt"
        )
        goal_block = pddl.get_goal_state_from_preconditions(
            agent_action, self.pddl_domain_path
        )
        pddl.create_pddl_problem_file(
            goal_block,
            self.precondition_pddl_problem_path,
            updated_domain_path,
            self.env,
        )

        # determine if those preconditions are satisfied
        preconds_verified = pddl.check_if_goal_state_satisfied(
            self.precondition_pddl_problem_path,
            updated_domain_path,
            self.solver,
        )

        if preconds_verified:
            self.react_proposed_w_satisfied_preconditions += 1
            self.logger.info(f"{Fore.GREEN}All preconditions satisfied...{Style.RESET_ALL}")
            return True

        # if the preconditions for the action are not satisfied, attempt to
        # satisfy those preconditions with the classical planner
        self.logger.info(f"{Fore.RED}Preconditions not satisfied! Attempting to satisfy with classical planner...{Style.RESET_ALL}")
        self.react_proposed_w_unsatisfied_preconditions += 1
        precondition_plan_result_flag = pddl.create_plan(
            self.precondition_plan_path,
            updated_domain_path,
            self.precondition_pddl_problem_path,
            self.solver,
        )
        if precondition_plan_result_flag != 0:
            # no plan could be generated
            self.precondition_fails_to_generate_plan += 1
            self.logger.info(
                "Unable to generate valid plan to satisfy preconditions. Will let ReAct take care of it..."
            )
            return False
        
        # valid plan generated by classical planner
        if not os.path.exists(self.precondition_plan_path):
            raise Exception("A plan file should have been generated.")
        
        self.precondition_valid_plan_generated += 1
        self.logger.info(
            "Some preconditions not satisfied, attempting to satisfy them with classical planner..."
        )
        cp_result_flag, obs = await self.cp.enact_plan(
            self.precondition_plan_path,
            self.solver,
            "classical - precondition_satisfy",
        )
        os.remove(self.precondition_plan_path)

        if cp_result_flag == 0:
            self.logger.info("Successfully satisfied preconditions.")
            self.precondition_plan_enacted_successfully += 1
            return True
        
        self.precondition_plan_not_enacted_successfully += 1
        self.logger.info(
            "There was an error while enacting the preconditions satifying plan. Will let ReAct take care of it..."
        )
        return False

    def parse_domain_predicate_info(self, domain_file_path: str) -> str:
        """Returns a list of strings representing the predicate attributes
        defined in a .pddl domain file."""
        predicates_block = get_pddl_block_from_pddl_file(
            domain_file_path, "(:predicates"
        )
        # remove comments for token efficiency
        predicates_block = re.sub(r";.*(?=\n)", "", predicates_block)
        return predicates_block

    def _create_updated_domain_file(self) -> str:
        # load the old domain file
        with open(self.pddl_domain_path, "r") as file:
            initial_domain = file.read()

        initial_types_block: str = get_pddl_block_from_pddl_file(
            self.pddl_domain_path, "(:types"
        )
        types_block_list = initial_types_block.split("\n")
        new_types = set()
        for obj_mdata in self.env.obj_db.get_obj_metadata_list():
            if "ai2thor" in self.env.__class__.__name__.lower():
                pddl_type = "receptacle" if obj_mdata["receptacle"] else "object"
            elif "alfworld" in self.env.__class__.__name__.lower():
                # pddl_type = "receptacle" if obj_mdata["isReceptacle"] else "object"
                pddl_type = "object"
            else:
                raise Exception("Invalid environment.")
            # reset the Butterknife type to just be the Knife type
            if "knife" in obj_mdata["objectType"].lower():
                obj_mdata["objectType"] = "Knife"
            new_types.add("\t\t" + obj_mdata["objectType"].lower() + f" - {pddl_type}")
        [types_block_list.insert(-1, x) for x in new_types]
        new_types_block = "\n".join(types_block_list)
        new_domain = initial_domain.replace(initial_types_block, new_types_block)

        temp_domain_path = os.path.join(
            os.path.dirname(self.pddl_problem_file_path), "temp_domain.pddl"
        )
        with open(temp_domain_path, "w") as file:
            file.write(new_domain)

        return temp_domain_path

    def save_results(self, task_metadata: TaskMetadata):

        total_token_count = self.get_total_token_count()
        action_sequence_length = self.get_action_sequence_length()
        env_steps = self.get_env_steps()
        self.log_env_steps()

        result_save_path = os.path.join(self.logging_dir, "results.json")

        is_goal_state_reached = self.env.check_if_goal_state_reached(task_metadata)

        num_classical_proposed_actions = len([x for x in self.state.action_call_history.history if x.action_type == "classical"])
        num_classical_proposed_actions_successful = len([x for x in self.state.action_call_history.history if (x.action_type == "classical" and "Successfully" in x.observation)])
        num_precondition_proposed_actions = len([x for x in self.state.action_call_history.history if x.action_type == "classical - precondition_satisfy"])
        num_precondition_proposed_actions_successful = len([x for x in self.state.action_call_history.history if (x.action_type == "classical - precondition_satisfy" and "Successfully" in x.observation)])
        num_react_proposed_actions = len([x for x in self.state.action_call_history.history if x.action_type == "react"])
        num_react_proposed_actions_successful = len([x for x in self.state.action_call_history.history if (x.action_type == "react" and "Successfully" in x.observation)])

        if not self.use_classical_planner:
            num_classical_proposed_actions = "n/a"
            num_classical_proposed_actions_successful = "n/a"

        if not self.use_precondition_verification:
            num_precondition_proposed_actions = "n/a"
            num_precondition_proposed_actions_successful = "n/a"

        results = {
            "Task Success": is_goal_state_reached,
            "Token Count": total_token_count,
            "Env Steps": env_steps,
            "Action Sequence Length": action_sequence_length,
            "ReAct Proposed w Unsatisfied Preconditions": self.react_proposed_w_unsatisfied_preconditions,
            "ReAct proposed w Satisfied Preconditions": self.react_proposed_w_satisfied_preconditions,
            "Precondition Fails to Generate Plan": self.precondition_fails_to_generate_plan,
            "Precondition Valid Plan Generated": self.precondition_valid_plan_generated,
            "Precondition Plan Enacted Successfully": self.precondition_plan_enacted_successfully,
            "Precondition Plan Not Enacted Successfully": self.precondition_plan_not_enacted_successfully,
            "Global Planner Fails to Generate Plan": self.global_planner_fails_to_generate_plan,
            "Global Planner Generates Plan": self.global_planner_generates_plan,
            "Global Planner Plan Not Enacted Successfully": self.global_planner_plan_not_enacted_successfully,
            "Global Planner Plan Enacted Successfully": self.global_planner_plan_enacted_successfully,
            "Generated Goal State Correct": self.generated_goal_state_correct,
            "Num Classical Proposed Actions": num_classical_proposed_actions,
            "Num Classical Proposed Actions Successful": num_classical_proposed_actions_successful,
            "Num Precondition Proposed Actions": num_precondition_proposed_actions,
            "Num Precondition Proposed Actions Successful": num_precondition_proposed_actions_successful,
            "Num ReAct Proposed Actions": num_react_proposed_actions,
            "Num ReAct Proposed Actions Successful": num_react_proposed_actions_successful,
        }

        if "ai2thor" in self.env.__class__.__name__.lower():
            # save the final environment state for manual verification of task
            # completion
            save_file = os.path.join(self.logging_dir, "final_env_state.json")
            with open(save_file, "w") as file:
                json.dump(self.env.obj_db.obj_metadatas, file, indent=3)

            # create and save a video
            log_file_path = None
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file_path = handler.baseFilename  # The path to the log file
            if log_file_path is None:
                raise Exception(
                    "Log file path could not be found via the logger object."
                )
            actions = list()
            for action in self.state.action_call_history.history:
                action_input = str(action.llm_action_input)
                action_name = action.llm_action_name
                actions.append(f"{action_name}({action_input})")
            # save the top down videos
            top_view_img_path = os.path.join(self.logging_dir, "top_view_imgs")
            front_view_img_path = os.path.join(self.logging_dir, "front_view_imgs")
            create_task_media(
                img_folder_path=top_view_img_path,
                log_file_path=log_file_path,
                output_vid_path=os.path.join(self.logging_dir, "top_view_task.mp4"),
                output_gif_path=os.path.join(self.logging_dir, "top_view_task.gif"),
                actions=actions,
            )
            # save the front facing videos
            create_task_media(
                img_folder_path=front_view_img_path,
                log_file_path=log_file_path,
                output_vid_path=os.path.join(self.logging_dir, "front_view_task.mp4"),
                output_gif_path=os.path.join(self.logging_dir, "front_view_task.gif"),
                actions=actions,
            )

            # remove the imgs folder
            shutil.rmtree(top_view_img_path)
            shutil.rmtree(front_view_img_path)

        with open(result_save_path, "w") as file:
            json.dump(results, file, indent=3)

        # save the task information
        with open(os.path.join(self.logging_dir, "task_description.json"), "w") as file:
            if self.global_task_metadata is not None:
                json.dump(
                    self.global_task_metadata.get_task_information(), file, indent=3
                )
            else:
                json.dump(self.task_metadata.get_task_information(), file, indent=3)

        # save the action information
        with open(
            os.path.join(self.logging_dir, "action_sequence_verbose.json"), "w"
        ) as file:
            history = [x.get_action_info_dict() for x in self.state.action_call_history.history]
            json.dump(history, file, indent=3)

        with open(os.path.join(self.logging_dir, "action_sequence.txt"), "w") as file:
            for action in self.state.action_call_history.history:
                file.write(
                    f"{action.llm_action_name}({str(action.llm_action_input)}) - {action.action_type}\n"
                )

    def _get_ai2thor_action_from_human_input(self) -> AgentAction:
        """Prompts the user to provide an action. This is used for debugging purposes
        as the AI2Thor simulator is riddled with bugs and we want to ensure that we
        take care of these bugs before running experiments and collecting metrics."""
        # get a list of actions that are available to the agent
        action_names = [x.llm_action_name for x in self.actions]

        correct_action_name = False
        while not correct_action_name:
            chosen_action_name = input(f"Please select an action from [{', '.join(action_names)}]: ")
            if chosen_action_name not in action_names:
                print(f"Invalid Action Name... Try again.")
            else:
                correct_action_name = True
        
        # prompt the user for the action arguments
        chosen_action_schema = [x for x in self.actions if x.llm_action_name == chosen_action_name][0]
        action_args = list(chosen_action_schema.llm_arg_desc.keys())

        chosen_action_user_args = dict()
        for action_arg in action_args:
            valid_arg = False
            while not valid_arg:
                action_arg_val = input(f"Input the arg value for {action_arg}: ")
                if action_arg_val not in self.env.obj_db.llm_ids:
                    print(f"Input arg not a valid object id.")
                else:
                    valid_arg = True
                    chosen_action_user_args[action_arg] = action_arg_val
        
        # get the AgentAction from the user input
        chosen_action_pddl_action_name = get_pddl_action_name(
            chosen_action_name, self.actions
        )
        chosen_action_pddl_action_input = get_pddl_action_input(
            chosen_action_name, 
            chosen_action_user_args, 
            self.actions, 
            self.state, 
            self.env
        )

        return AgentAction(
            "human-user",
            chosen_action_schema.action_function,
            self.agent_id,
            "This is a human user-chosen action.",
            chosen_action_name,
            chosen_action_user_args,
            chosen_action_pddl_action_name,
            chosen_action_pddl_action_input
        )

class GoalStateLLM:
    def __init__(
        self, llm: str, env: Environment, logger: logging.Logger, max_runs: int
    ):
        self.llm = get_llm(llm)
        self.env = env
        self.logger = logger
        self.max_runs = max_runs
        self.token_count = 0

    def reset(self):
        self.token_count = 0

    def get_pddl_goal_block(
        self,
        task_metadata: TaskMetadata,
        domain_predicate_info: str,
        goal_generation_examples: str,
    ):
        input = {
            "input_task": task_metadata.task_natural_language,
            "discovered_objects": ", ".join([i for i in self.env.obj_db.llm_ids]),
            "goal_generation_examples": goal_generation_examples,
            # "goal_state_helper_statements": "\n".join(
            #     [v["desc"] for _, v in get_goal_state_helper_descs().items()]
            # ),
            "domain_predicate_info": domain_predicate_info,
        }
        if "alfworld" in self.env.__class__.__name__.lower():
            del input["discovered_objects"]  # alfworld does not need this
            prompt = GOAL_STATE_TEMPLATE_ALFWORLD.format(**input)
        else:
            prompt = GOAL_STATE_TEMPLATE_AI2THOR.format(**input)
        if COPY_LLMDP_ALFWORLD_GOAL_PROMPT and "alfworld" in self.env.__class__.__name__.lower():
            prompt = GENERATE_GOAL_PROMPT_LLMDP + [
                {
                    "role": "user",
                    "content": AlfworldEnv.task_natural_language
                }
            ]
        goal_states = dict()

        # maybe it is more robust if we run the model many times and
        # remove outlier goal states...
        high_level_goal_state = None
        for _ in range(self.max_runs):
            if COPY_LLMDP_ALFWORLD_GOAL_PROMPT and "alfworld" in self.env.__class__.__name__.lower():
                llm_output = self.llm.invoke(messages=prompt)
            else:
                llm_output = self.llm.invoke(prompt)
            self.token_count += llm_output.token_count
            pddl_goal_block = self._parse_output(llm_output.content)
            if pddl_goal_block != "Could not parse output.":
                break

        if pddl_goal_block == "Could not parse output.":
            return f"Goal state could not be parsed after {self.max_runs} tries."
            raise Exception(
                f"Goal state could not be parsed after {self.max_runs} tries."
            )

        return pddl_goal_block
        #     high_level_goal_state = self._parse_output(llm_output.content)
        #     # self.logger.info(f"High level goal state: {high_level_goal_state}")
        #     high_level_goal_state = high_level_goal_state.replace(" ", "")

        #     # Use regex to match function calls
        #     pattern = r"[a-zA-Z_]+\([^\)]*\)"
        #     goal_state_list = re.findall(pattern, high_level_goal_state)
        #     for goal_state in goal_state_list:
        #         if goal_state in goal_states.keys():
        #             goal_states[goal_state] += 1
        #         else:
        #             goal_states[goal_state] = 1

        # assert high_level_goal_state is not None

        # # self.logger.info(f"Goal state counts: {goal_states}")
        # self.logger.info(f"Goal states: {[x for x in goal_states.keys()]}")

        # pddl_goal_block = self._generate_pddl_goal_block(high_level_goal_state)
        # return pddl_goal_block

    def _parse_output(self, llm_output: str) -> str:
        if COPY_LLMDP_ALFWORLD_GOAL_PROMPT and "alfworld" in self.env.__class__.__name__.lower():
            return llm_output.replace("```", "").replace("lisp", "")
        try:
            llm_output = llm_output.replace("*", "")
            start_idx = llm_output.find("Goal State:")

            if start_idx == -1:
                # the Goal State: substring was not found, therefore the LLM formatted
                # its output incorrectly
                exception_return_msg = (
                    "Goal State: substring not found. LLM formatted output incorrectly."
                )
                raise Exception(exception_return_msg)

            llm_output = llm_output[start_idx:]

            start_idx = llm_output.find("(") - 1
            end_idx = llm_output.find("Done!")

            if start_idx == -1 or end_idx == -1:
                # the goal state list brackets were not found, therefore the LLM formatted
                # its output incorrectly
                exception_return_msg = "Could not parse output."
                raise Exception(exception_return_msg)

            llm_output = llm_output[start_idx + 1 : end_idx]
            llm_output.replace("'", "").replace('"', "").replace("pddl", "")

            return llm_output
        except:
            exception_return_msg = "Could not parse output."
            return exception_return_msg

    def _generate_pddl_goal_block(self, high_level_goal_state: str) -> List[str]:
        """Takes in a high level goal state produced by the goal state
        generating LLM and produces a valid pddl goal state block.

        Args:
            high_level_goal_state (str): a string of function calls to high
                level goal state helper functions in the form:
                    function_name(arg_1, ..., arg_N)

        Returns:
            pddl_goal_block (List[str]): a list of strings that when saved to
                 a problem.pddl file with file.writelines(pddl_goal_block) will save
                 a :goal block in the pddl file.
        """
        high_level_goal_state = high_level_goal_state.replace(" ", "")
        pddl_goal_states = list()

        # Use regex to match function calls
        pattern = r"[a-zA-Z_]+\([^\)]*\)"
        goal_state_list = re.findall(pattern, high_level_goal_state)

        goal_state_helper_descriptions = get_goal_state_helper_descs()
        for goal_state in goal_state_list:
            func_name, args = formatting.process_func_in_str_format(goal_state)
            func = goal_state_helper_descriptions[func_name]["func"]
            arg_types = goal_state_helper_descriptions[func_name]["arg_types"]
            if len(arg_types) != len(args):
                raise Exception(
                    "Mismatched args and arg types. Did the LLM format args incorrectly?"
                )
            # attempt to parse arguments to their target types
            try:
                args = [arg_types[idx](i) for idx, i in enumerate(args)]
            except Exception as error:
                raise error
            # TODO: need to break this or wrap into the actual description of
            # the functions..
            pddl_goal_states.extend(func(self.env.obj_db, *args))

        # format the list of pddl_goal_states into a valid pddl goal block
        pddl_goal_block = formatting.format_pddl_goal_state(pddl_goal_states)
        return pddl_goal_block
