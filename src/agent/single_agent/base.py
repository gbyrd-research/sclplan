"""
Defines the Base ConceptAgent agent from which all other agents
will inherit from.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import shutil

from src.tasks.base import TaskMetadata
from src.utils.misc.logging_utils import ensure_dir_exists, get_hydra_run_dir


class BaseAgent(ABC):

    def __init__(
            self, 
            agent_id: int,
            env,
            results_dir: str
        ):
        self.agent_id = agent_id

        # set the logger name
        self.logger_name = f"agent_{self.agent_id} - {self.__class__.__name__}"
        self.logger = logging.getLogger(self.logger_name)
        self.env = env

        self.results_dir = results_dir
        if self.results_dir is not None:
            # need to copy hydra config into folder
            if not os.path.exists(os.path.join(self.results_dir, ".hydra")):
                shutil.copytree(
                    os.path.join(get_hydra_run_dir(), ".hydra"), 
                    os.path.join(self.results_dir, ".hydra")
                )
        else:
            self.results_dir = get_hydra_run_dir()

    @abstractmethod
    def save_results(self, task_metadata: TaskMetadata):
        """Get the results from the agent's attempt to complete the task."""

        pass

        # total_token_count = self.get_total_token_count()
        # action_sequence_length = self.get_action_sequence_length()
        
        # result_save_path = os.path.join(self.logging_dir, "results.json")

        # is_goal_state_reached = self.env.check_if_goal_state_reached()
        
        # results = {
        #     "is_goal_state_reached": is_goal_state_reached,
        #     "total_token_count": total_token_count,
        #     "action_sequence_length": action_sequence_length
        # }
        
        # with open(result_save_path, "w") as file:
        #     json.dump(results, file, indent=3)

        # # create a pddl problem file that represents the ground truth of
        # # the environment and goal state for the task
        # temp_pddl_goal_state_verification_path = "temp_result_pddl_file.pddl"
        
        # # ensure the ai2thor environment is fully populated with ground truth
        # # environment state
        # asyncio.run(AI2ThorEnv.populate_obj_db_w_ground_truth())
        
        # create_pddl_problem_file(
        #     task_metadata.pddl_goal_state,
        #     temp_pddl_goal_state_verification_path,
        #     AI2THOR_PDDL_DOMAIN_PATH,
        #     AI2ThorEnv()
        #     )
        
        # # determine if the overall goal state was reached
        # is_goal_state_reached = check_if_goal_state_satisfied(temp_pddl_goal_state_verification_path, AI2THOR_PDDL_DOMAIN_PATH)
        
        # # delete the temporary pddl file
        # os.remove(temp_pddl_goal_state_verification_path)
        
        # total_token_count = self.get_total_token_count()
        # action_sequence_length = self.get_action_sequence_length()
        
        # result_save_path = os.path.join(self.logging_dir, "results.json")
        
        # results = {
        #     "is_goal_state_reached": is_goal_state_reached,
        #     "total_token_count": total_token_count,
        #     "action_sequence_length": action_sequence_length
        # }
        
        # with open(result_save_path, "w") as file:
        #     json.dump(results, file, indent=3)

    @abstractmethod
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
        pass

    ############################################################################
    # abstract methods for collecting results

    @abstractmethod
    def get_total_token_count(self) -> int:
        pass

    @abstractmethod
    def get_action_sequence_length(self) -> int:
        pass

    def _set_logging_dir(
        self, 
        task_metadata: TaskMetadata, 
        global_task_metadata: TaskMetadata | None, 
        parent_dir: str = None
    ) -> None:

        if parent_dir is None:
            # set the logger dir and ensure that the file and parent
            # dirs exist
            hydra_output_dir = get_hydra_run_dir()
            if global_task_metadata is None:
                self.logging_dir = os.path.join(hydra_output_dir, task_metadata.task_name)
            else:
                self.logging_dir = os.path.join(
                    hydra_output_dir, global_task_metadata.task_name
                )
        else:
            if global_task_metadata is None:
                self.logging_dir = os.path.join(parent_dir, task_metadata.task_name)
            else:
                self.logging_dir = os.path.join(
                    parent_dir, global_task_metadata.task_name
                )
        if os.path.exists(os.path.join(self.logging_dir, "results.json")):
            raise Exception("You are about to delete a task that has generated results. You can remove this exception if you are sure this is what you want to do.")
        # clear out the logging directory
        if os.path.exists(self.logging_dir):
            shutil.rmtree(self.logging_dir)
        ensure_dir_exists(self.logging_dir)
        self.logfile = os.path.join(self.logging_dir, self.logger_name + ".log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # remove existing file handlers
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()
        

        new_file_handler = logging.FileHandler(self.logfile)
        new_file_handler.setFormatter(formatter)
        self.logger.addHandler(new_file_handler)
