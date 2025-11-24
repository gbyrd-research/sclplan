"""
Defines a React agent for use in Single Agent tasks in AI2Thor.
"""

import asyncio
from typing import Dict, Optional, Tuple

from src.agent.single_agent.base import BaseAgent
from src.tasks.base import TaskMetadata
from src.utils.agent.action import TaskCompletionAction
from src.utils.agent.react_w_pddl import ReactAgentWithPDDL
from src.utils.base_utils import get_obj_from_str
from src.utils.state.agent_state import AgentState


class ReactAgent(BaseAgent):

    def __init__(
        self,
        llm: str,
        env_target: str,
        react_action_schemas_target: str,
        agent_id: int,
        max_planning_steps: int = 25,
        llm_kwargs: Dict = {},
    ):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.env = get_obj_from_str(env_target)
        self.actions = get_obj_from_str(react_action_schemas_target)
        self.state = AgentState(agent_id)
        self.agent = ReactAgentWithPDDL(
            agent_id=agent_id,
            env=self.env(),
            llm=llm,
            actions=self.actions,
            state=self.state,
            llm_kwargs=llm_kwargs,
        )
        self.max_planning_steps = max_planning_steps
        self.llm_name = llm

    def get_total_token_count(self) -> int:
        total_token_count = 0
        for llm_response in self.agent.model_response_list:
            total_token_count += llm_response.token_count
        return total_token_count

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

        final_answer = await self._command(task_metadata.task_natural_language)
        agent_state_in_dict_format = self.state.get_agent_state_default()
        return (self.agent_id, final_answer, agent_state_in_dict_format)

    async def _command(self, natural_language_task: str) -> str:
        """Calls the agent executor and provides it with the
        initial natural language task to complete."""
        # briefly pause and let other functions run asynchronously
        await asyncio.sleep(0)
        for _ in range(self.max_planning_steps):
            agent_action = self.agent.get_next_action(natural_language_task)
            action_return_flag, obs = await agent_action.call_action()

            self.logger.info(
                "ReAct Predicted Action:\n" + agent_action.get_action_info_string()
            )

            self.state.action_call_history.add_action(agent_action)
            if isinstance(agent_action, TaskCompletionAction):
                return obs
        return "Task Could Not Be Completed: Max planning steps exceeded!"
