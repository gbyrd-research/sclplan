"""
Defines the MasterPlanner class for multi-agent planning.
"""

import os
from typing import Dict, List

from langchain.agents import AgentExecutor, Tool
from langchain_core.runnables.base import Runnable
from omegaconf import DictConfig

from agent.utils.logging_utils import ensure_dir_exists, get_hydra_run_dir
from src.agent.multi_agent.master_planner.prompt import (
    STAGE_1_INPUT_VARS,
    STAGE_1_SYS_PROMPT,
    STAGE_2_INPUT_VARS,
    STAGE_2_SYS_PROMPT,
    Stage1OutputParser,
    Stage1PromptTemplate,
    Stage2OutputParser,
    Stage2PromptTemplate,
)
from src.agent.single_agent.react.agent import ReactAgentWithPDDL
from src.agent.tool_engine.ai2thor_tool_engine import AI2ThorToolEngine
from src.agent.tool_engine.master_planner_tool_engine import MasterPlannerToolEngine
from src.agent.utils.build_agent import (
    get_agent,
    get_agent_executor,
    get_basic_runnable,
)
from src.tasks.base import TaskMetadata


class MasterPlanner:
    """The Master Planner is the highest level planner for multi-agent
    robotic planning. It takes in a high level task and a description
    of the availabe agents and decomposes that high level tasks into
    subtasks that it then delegates to the agents in the multi-agent
    team.
    """

    def __init__(
        self,
        cfg_MP: DictConfig,  # master planner config
        cfg_BA: DictConfig,  # base agent config
        cfg_AI2THOR_TE: DictConfig,  # ai2thor tool engine config
        agent_count: int,  # num agents in the multi-agent team
    ):
        self.agent_count = agent_count
        self.agents = self._create_multi_team_agents(
            cfg_AI2THOR_TE, cfg_BA, agent_count
        )
        self.tool_engine = MasterPlannerToolEngine(self.agents)
        self.stage_1 = self._build_stage_1_runnable(cfg_MP)
        self.stage_2 = self._build_stage_2_agent_executor(cfg_MP)
        self.cfg = cfg_MP

    def perform_task(self, task_metadata: TaskMetadata):
        self.tool_engine.reset(task_metadata)
        self.task = task_metadata.task_natural_language
        self.team_cfg = task_metadata.team_cfg
        self._setup_loggers(task_metadata)

        # run stage 1 to get the initial predicted sequence of actions
        disc_objs = ", ".join(AI2ThorToolEngine.objects.llm_ids)
        stage_1_output = self.stage_1.invoke(
            {"discovered_objects": disc_objs, "task_natural_language": self.task}
        )
        actionable_plan = self.stage_1_output_parser.parse(stage_1_output.content)
        if actionable_plan == "Could not parse output.":
            raise Exception("Need to write code to handle this.")

        # run stage 2 to decompose sequence of actions to subtasks and
        # delegate to agents in the multi-agent team
        stage_2_output = self.stage_2.invoke(
            {
                "multi_agent_team_overview": str(self.team_cfg),
                "task_natural_language": self.task,
                "discovered_objects": disc_objs,
                "actionable_plan": actionable_plan,
            }
        )
        return stage_2_output

    def _setup_loggers(self, task_metadata: TaskMetadata) -> None:
        """Sets up the log files for the Stage 1 and Stage 2 agents."""
        hydra_output_dir = get_hydra_run_dir()
        self.logger_dir = os.path.join(hydra_output_dir, task_metadata.task_name)
        ensure_dir_exists(self.logger_dir)
        stage1_logfile = os.path.join(self.logger_dir, "MasterPlannerStage1.log")
        stage2_logfile = os.path.join(self.logger_dir, "MasterPlannerStage2.log")
        self.stage1_prompt.set_logfile(stage1_logfile)
        self.stage2_prompt.set_logfile(stage2_logfile)

    def _create_multi_team_agents(
        self, cfg_TE: DictConfig, cfg_BA: DictConfig, agent_count: int
    ) -> List[ReactAgentWithPDDL]:
        """Creates each agent in the multi-agent team."""
        # instantiate the ai2thor controller for the ToolEngine class
        AI2ThorToolEngine.initialize_ai2thor_controller(cfg_TE, agent_count)
        # create a list of AI2ThorAgents
        agents = list()
        for agent_id in range(agent_count):
            # agents indexed in the list by their agent id :)
            agents.append(ReactAgentWithPDDL(cfg_BA, agent_id))
        return agents

    def _get_all_agent_actions(self) -> List[Tool]:
        """Loop through all agents in self.agents and include all possible
        action from all of the agents."""
        agent_actions: Dict[str, Tool] = dict()
        for agent in self.agents:
            for tool in agent.actions:
                if (
                    tool.name in agent_actions
                    and tool.description != agent_actions[tool.name].description
                ):
                    raise ValueError(
                        "Same tool name but different descriptions. All tools with the same name must be the same tool."
                    )
                agent_actions[tool.name] = tool
        return list(agent_actions.values())

    def _build_stage_1_runnable(self, cfg_MP) -> Runnable:
        """Builds the Master Planner agent stage 2 executor."""
        self.stage_1_template = STAGE_1_SYS_PROMPT
        self.stage_1_input_vars = STAGE_1_INPUT_VARS
        agent_actions = [
            f"{x.name}: {x.description}" for x in self._get_all_agent_actions()
        ]
        agent_action_names = ", ".join([x.name for x in self._get_all_agent_actions()])
        self.partial_vars = {
            "agent_actions": agent_actions,
            "agent_action_names": agent_action_names,
        }
        self.stage_1_output_parser = Stage1OutputParser()
        runnable = get_basic_runnable(
            self.stage_1_template,
            cfg_MP.llm,
            self.stage_1_input_vars,
            self.partial_vars,
            Stage1PromptTemplate,
        )
        self.stage1_prompt: Stage1PromptTemplate = runnable.first
        return runnable

    def _build_stage_2_agent_executor(self, cfg_MP) -> AgentExecutor:
        """Builds the Master Planner agent stage 2 executor."""
        self.stage2_template = STAGE_2_SYS_PROMPT
        self.stage2_input_vars = STAGE_2_INPUT_VARS
        self.tools = self._get_tools()
        self.stage2_prompt = Stage2PromptTemplate(
            template=self.stage2_template,
            master_planner_actions=self.tools,  # type:ignore
            input_variables=self.stage2_input_vars,
        )
        self.output_parser = Stage2OutputParser()
        self.output_parser.interrupt = False
        agent = get_agent(
            cfg_MP.llm,
            self.stage2_prompt,
            self.tools,
            self.output_parser,
            cfg_MP.llm_agent_cfg,
        )
        return get_agent_executor(agent, self.tools, cfg_MP.max_planning_steps)

    def _get_tools(self) -> List[Tool]:
        delegate_subtasks_tool_description = 'Send subtasks to agents for completion. The input format is a dictionary with a single key: "subtask_delegations". The value of the "subtask_delegations" key is a dictionary that maps agent ids to natural language subtasks. Strictly follow the following format for inputting arguments to the Delegate Substasks tool: {"subtask_delegations": {"agent_0": "subtask in natural language", "agent_id_1": "subtask in natural language"}}. Do not send more than one subtask to the same agent at one time.'
        # delegate_subtasks_tool_description = 'Send subtasks to agents for completion. The input format is a dictionary with a single key: "subtask_delegations". The value of the "subtask_delegations" key is a dictionary that maps agent ids to natural language subtasks. Strictly follow the following format for inputting arguments to the Delegate Substasks tool: {"subtask_delegations": {"agent_0": "subtask in natural language", "agent_id_1": "subtask in natural language"}}.'
        tools = [
            Tool(
                name="Delegate Subtasks",
                func=self.tool_engine.delegate_subtasks,
                # strip newline characters from desc to avoid confusing model
                description=delegate_subtasks_tool_description.replace("\n", ""),
            )
        ]
        return tools
