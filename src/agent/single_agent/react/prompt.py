import logging
import re
import time
from typing import Any, List, Optional, Union

from langchain.agents import AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.tools import BaseTool
from pydantic import ConfigDict, Field


class MyAgentFinish(AgentFinish):
    tool: str = ""
    tool_input: str = ""

    def __init__(self, return_values, log):
        super().__init__(return_values, log)
        self.tool = "Task Completed"
        self.tool_input = ""


class BaseAgentPromptTemplate(BaseChatPromptTemplate):
    """Custom prompt template for agent that includes some ~ hacky ~ logging with environment variables.
    Should fix this at some point..
    """

    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    logger: logging.Logger | None = None

    def set_logfile(self, name: str, logfile: str):
        """Update the logfile dynamically."""
        self.logger = logging.Logger(name)

        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # add a new handler with the specified logfile
        handler = logging.FileHandler(logfile)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(handler)

    def format_messages(self, **kwargs) -> str:
        if self.logger is None:
            raise RuntimeError(
                "You must initialize the logger before you call the agent."
            )

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
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        formatted = self.template.format(**kwargs)
        self.logger.info("+" * 100 + "\n")
        self.logger.info(formatted)
        return [HumanMessage(content=formatted)]


class BaseAgentOutputParser(AgentOutputParser):
    """Custom output parser for agent that includes some ~ hacky ~ logging with environment variables.
    Should fix this at some point..
    """

    interrupt: bool = True
    synth_pub: Any = None
    final_answer: str = ""
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger("default"))
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_logfile(self, name: str):
        """Update the logfile dynamically."""
        self.logger = logging.Logger(name)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish, None]:
        if self.logger is None:
            raise RuntimeError(
                "You must initialize the logger before you call the agent."
            )
        self.logger.info(llm_output)
        self.logger.info("+" * 100 + "\n")
        print(f"Initial LLM output: {llm_output}")
        time.sleep(1.0)
        if self.interrupt:
            self.final_answer = "Halt requested."
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
            # self.logger.info(self.final_answer)
            return MyAgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": output},
                log=llm_output,
            )
        regex_thought = r"(.*?)[\n]*Action:[\s]*(.*)"
        match = re.search(regex_thought, llm_output, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            thought = thought.replace("Thought:", "")
        # get everything from Action: to \n and then from Action Input: to \n
        regex = r"Action:\s*(.*?)\n.*?Action Input:\s*(.*?)\n"
        match = re.search(regex, llm_output)
        action = None
        if match:
            action = match.group(1)
            action_input = match.group(2)
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        # if not match:
        #     return "The tool calling format is invalid. Please try again and format correctly."
        if action is not None:
            action = match.group(1).strip()
            action_input = match.group(2)
        else:
            action_input = "None"
        return AgentAction(
            tool=action.strip(" "),
            tool_input=action_input.strip(" ").strip('"'),
            log=llm_output,
        )


# below are the various prompting templates that will be used for experiments
template_1_input_variables = ["input", "discovered_objects", "intermediate_steps"]

template_1 = """You are a robot who can use tools to take action in the real world.  You have access to the following tools:

{tools}

Strictly use the following format:

Task: the task you must complete
Thought: you should always think about what to do and replan if needed
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have successfully completed my task
Task Complete: Provide your answer justifying why the task has been completed.

Note: Make sure that you only complete the initial Goal. Please do not do anything extra. Ensure that you only do exactly what is asked of you and no more.

Note: Do not even propose "None" action. Instead, say 'Task Complete':.

Begin!

The following objects have been discovered in the environment: {discovered_objects}

Task: {input}
{agent_scratchpad}"""


TEMPLATES = {1: template_1}
INPUT_VARS = {1: template_1_input_variables}
