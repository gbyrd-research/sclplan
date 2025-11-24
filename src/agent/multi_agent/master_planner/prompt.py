"""
Prompts for the Master Planner.
"""

import json
import logging
import re
from typing import Any, List, Tuple, Union

from langchain.agents import AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, PromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.tools import BaseTool
from langchain_core.output_parsers import BaseOutputParser
from pydantic import ConfigDict, Field


class MyAgentFinish(AgentFinish):
    tool: str = ""
    tool_input: str = ""

    def __init__(self, return_values, log):
        super().__init__(return_values, log)
        self.tool = "Task Completed"
        self.tool_input = ""


MASTER_PLANNER_STAGE1_LOGGER_NAME = "MasterPlannerStage1"
MASTER_PLANNER_STAGE2_LOGGER_NAME = "MasterPlannerStage2"


class Stage1PromptTemplate(PromptTemplate):
    """Custom prompt template for agent that includes some ~ hacky ~ logging with environment variables.
    Should fix this at some point..
    """

    # The template to use
    template: str
    # The list of tools available
    # master_planner_actions: List[BaseTool]
    logger: logging.Logger = logging.getLogger(MASTER_PLANNER_STAGE1_LOGGER_NAME)

    def set_logfile(self, logfile: str):
        """Update the logfile dynamically."""
        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # add a new handler with the specified logfile
        handler = logging.FileHandler(logfile)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(handler)

    def format(self, **kwargs) -> str:
        partial_and_input = kwargs | self.partial_variables
        formatted = self.template.format(**partial_and_input)
        self.logger.info(formatted)
        return formatted


class Stage2PromptTemplate(BaseChatPromptTemplate):
    """Custom prompt template for agent that includes some ~ hacky ~ logging with environment variables.
    Should fix this at some point..
    """

    # The template to use
    template: str
    # The list of tools available
    master_planner_actions: List[BaseTool]

    logger: logging.Logger = logging.getLogger(MASTER_PLANNER_STAGE2_LOGGER_NAME)

    def set_logfile(self, logfile: str):
        """Update the logfile dynamically."""
        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # add a new handler with the specified logfile
        handler = logging.FileHandler(logfile)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(handler)

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        observations = []
        for action, observation in intermediate_steps:
            if isinstance(action, str):
                thoughts += "Make sure you output something!--"
                continue
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            observations.append(f"Observation: {observation}")

        # if len(observations) > 0:
        #     self.logger.info(observations[-1].replace("\n\n", "\n") + "\n")
        #     self.logger.info(
        #         (observations[-1].replace("\n\n", "\n")).split("|")[0] + "\n"
        #     )  # skip logging the scene information

        thoughts = thoughts[:-10]

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["master_planner_actions"] = "\n".join(
            [
                f"{master_planner_action.name}: {master_planner_action.description}"
                for master_planner_action in self.master_planner_actions
            ]
        )

        # Create a list of tool names for the tools provided
        kwargs["master_planner_action_names"] = ", ".join(
            [
                master_planner_action.name
                for master_planner_action in self.master_planner_actions
            ]
        )

        formatted = self.template.format(**kwargs)
        self.logger.info("+" * 100 + "\n")
        self.logger.info(formatted)
        return [HumanMessage(content=formatted)]


class Stage1OutputParser(BaseOutputParser[Tuple]):
    """Custom output parser for the filter by relevancy LLM."""

    interrupt: bool = True
    synth_pub: Any = None
    final_answer: str = ""
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger(MASTER_PLANNER_STAGE1_LOGGER_NAME)
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, llm_output: str) -> str:
        # cls.logger.info(f"MasterPlanner stage 1.")
        try:
            llm_output = llm_output.replace("*", "")
            # Regular expression to match the Action and the JSON part of Action Input
            pattern = r"Action:\s+(\w+)\nAction Input:\s+(\{.+?\})"

            # Parse the actions and inputs
            matches = re.findall(pattern, llm_output)

            # Process each action with its corresponding JSON
            formatted_actions = []
            for action, json_str in matches:
                # Load the JSON part to handle multiple keys dynamically
                action_input = json.loads(json_str)
                # Format each key-value pair within the parentheses, separated by commas
                input_details = ", ".join(
                    [f"{key}:{value}" for key, value in action_input.items()]
                )
                # Append the formatted string to the result list
                formatted_actions.append(f"{action}({input_details})")
            self.logger.info(f"LLM Generated Actionable Plan:\n")
            formatted_actions = "\n".join(formatted_actions)
            self.logger.info(formatted_actions)
            return formatted_actions
        except:
            exception_return_msg = "Could not parse output."
            self.logger.info(exception_return_msg)
            return exception_return_msg


class Stage2OutputParser(AgentOutputParser):
    """Custom output parser for the master planner."""

    interrupt: bool = True
    synth_pub: Any = None
    final_answer: str = ""
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger(MASTER_PLANNER_STAGE2_LOGGER_NAME)
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish, None]:
        # # assert os.environ['LANGCHAIN_PARSER_INTERRUPT'] == '0', "Error: It seems you have forgot to reset this environment variable to '0'"
        # print(f"Initial LLM output: {llm_output}")
        # if "Final Answer" in llm_output:
        #     self.logger.info((llm_output + "\n").replace("\n\n", "\n"))
        # else:
        #     self.logger.info(llm_output.replace("\n\n", "\n"))

        # log output
        self.logger.info(llm_output)
        self.logger.info("+" * 100 + "\n")

        if self.interrupt:
            self.final_answer = "Halt requested."
            self.logger.info(self.final_answer)
            return MyAgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": self.final_answer},
                log=llm_output,
            )

        # Check if agent should finish
        if "Task Complete:" in llm_output:
            output = llm_output.split("Task Complete:")[-1].strip()
            self.final_answer = output
            return MyAgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": output},
                log=llm_output,
            )

        # do some basic formatting to remove erroneous characters
        llm_output = llm_output.replace("*", "")

        regex_thought = r"(.*?)[\n]*Action:[\s]*(.*)"
        match = re.search(regex_thought, llm_output, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            thought = thought.replace("Thought:", "")

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        action = None
        if not match:
            regex = r"Action: (.*?)[\n]*"
            match = re.search(regex, llm_output, re.DOTALL)
            if match:
                action = re.split(regex, llm_output, 1)[-1].strip()

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            return "The tool calling format is invalid. Please try again and format correctly."
        if action is None:
            action = match.group(1).strip()
            action_input = match.group(2)
        else:
            action_input = "None"

        # Return the action and action input
        if "exit()" in action_input:
            return f"Error: Do not attempt to exit the python file."
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


STAGE_1_INPUT_VARS = ["discovered_objects", "task_natural_language"]

STAGE_2_INPUT_VARS = [
    "discovered_objects",
    "task_natural_language",
    "actionable_plan",
    "multi_agent_team_overview",
    "intermediate_steps",
]

STAGE_1_SYS_PROMPT = """You are an intelligent agent that serves as the 
high level planner for a multi-agent system. You have access to the following
actions:

{agent_actions}

Please provide an actionable plan by listing a sequence of actions from the
list of available actions to complete the high level goal. Strictly use the 
following format:

Goal: the goal you must generate a plan for
Actionable Plan:
Thought: the justification for choosing the next action
Action: the action to take, should be one of [{agent_action_names}]
Action Input: the input to the action
... (this Thought/Action/Action Input sequence should repeat N times until the plan is complete)
Plan Complete: Explain why you believe this sequence of actions will satisfy the goal

Begin!

The following objects have been discovered in the environment: {discovered_objects}

Goal: {task_natural_language}"""

STAGE_2_SYS_PROMPT = """You are an intelligent agent that serves as the 
high level planner for a multi-agent system. Your job is to receive a high 
level goal and an actionable plan to achieve that goal and decompose that
plan into several subtasks. You should then delegate these subtasks to
various agents in a multi-agent system in order to complete the subtasks
efficiently. To do this, you have access to the following actions:

{master_planner_actions}

Strictly use the following format:
Goal: the goal you must complete
Actionable Plan: an actionable plan to achieve the goal
Multi-agent Team Overview: The overview of the multi-agent team
Thought: You should always think about what to do and replan if needed
Task Decomposition: A list of decomposed subtasks created by grouping together actions from the actionable plan
Action: the action to take, should be one of [{master_planner_action_names}]
Action Input: the input to the action
Observation: The return observation from the action
... (this Thought/Task Decomposition/Action/Action Input/Observation can repeat N times)
Thought: The agents have successfully completed the subtasks and together have achieved the global task
Task Complete: Explain why you believe the task is completed

Begin!

The following objects have been discovered in the environment: {discovered_objects}

Global Task: {task_natural_language}
Actionable Plan: {actionable_plan}
Multi-agent Team Overview: {multi_agent_team_overview}
{agent_scratchpad}"""
