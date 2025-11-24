import inspect
import re
from typing import Callable, Dict, List, Optional
import logging

from src.utils.agent.action import AgentAction, TaskCompletionAction
from src.utils.classical_planner.pddl import (
    ReactActionSchema,
    get_pddl_action_input,
    get_pddl_action_name,
)
from src.utils.environment.base import Environment
from src.utils.llms.base import LLMResponse
from src.utils.llms.load import get_llm
from src.utils.state.agent_state import AgentState

# define parsing result integer flags
VALID_ACTION_PREDICTION = 0
VALID_TASK_COMPLETION_PRED = 1
INVALID_FORMATTING_REDO = 2
INVALID_FORMATTING_ACTION = 3
INVALID_FORMATTING_ACTION_KWARGS = 4


class ReactAgentWithPDDL:
    """A React agent that allows us to see each step of the process instead
    of abstracting everything away like Langchain does. Allows for finetuned
    control. Use this agent if you plan on using a PDDL defined classical
    planner to assist your agent.
    """

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        llm: str,
        actions: List[ReactActionSchema],
        state: AgentState,
        logger: logging.Logger,
        llm_kwargs: Dict = {}
    ):
        """
        Args:
            llm (str): the string name of the LLM you wish to build
            actions (List[ReactActionSchema]): a list of actions specified as
                ReacActionSchema objects
            template (Optional[str]): a custom React Template if you wish to add one?
                must have the same input vars as DEFAULT
        """
        self.agent_id = agent_id
        self.env = env
        # ensure the stop condition is not overrided
        if "stop" in llm_kwargs.keys():
            raise KeyError(
                "For react agent, do not override the stop criteria as this is done within the class."
            )
        self.llm = get_llm(llm, stop=["Observation"], **llm_kwargs)

        if "ai2thor" in env.__class__.__name__.lower():
            self.template = AI2THOR_DEFAULT_REACT_TEMPLATE
        else:
            self.template = DEFAULT_REACT_TEMPLATE
        self.actions = actions
        self.partial_vars = self._get_partial_vars()
        self.state = state
        self.logger = logger

        # if the LLM formats incorrectly, it will be prompted to
        # retry a max number of times
        self.max_retry_count = 5
        self.model_response_list: List[LLMResponse] = list()

    def reset(self):
        self.model_response_list = list()

    def get_next_action(self, task: str) -> AgentAction | TaskCompletionAction:
        """Given a task, ask the agent to predict the next action
        given the history of action calls.

        Args:
            task (str): the task represented in natural language

        Returns:
            action (Callable): the predicted action function to call
            kwargs (Dict): the predicted arguments for the action
                function
        """
        orig_history = self.state.action_call_history.get_history_llm_facing()

        input_dict = {
            "task": task,
            "discovered_objects": (", ").join(self.env.obj_db.llm_ids),
            "history": orig_history,
        }

        retry_count = 0
        while 1:

            if retry_count > self.max_retry_count:
                break

            llm_input_str = self.template.format(**input_dict, **self.partial_vars)

            # save out the initial system prompt if this is the first call to the
            # ReAct agent
            if len(self.model_response_list) == 0:
                self.logger.info(llm_input_str)

            model_response = self.llm.invoke(llm_input_str)

            self.model_response_list.append(model_response)

            result = self._parse_model_output(model_response.content)

            # check if model predicted the task was completed
            if result["parser_flag"] == VALID_TASK_COMPLETION_PRED:
                return TaskCompletionAction(self.agent_id)

            # check if the model predicted a valid action call
            if result["parser_flag"] == VALID_ACTION_PREDICTION:
                action = result["action"]
                action_input = eval(result["action_input"])
                action_schema = [
                    a for a in self.actions if a.llm_action_name == action
                ][0]
                pddl_action_name = get_pddl_action_name(action, self.actions)
                pddl_action_input = get_pddl_action_input(
                    action, action_input, self.actions, self.state, self.env
                )
                return AgentAction(
                    "react",
                    action_schema.action_function,
                    self.agent_id,
                    result["thought"],
                    action,
                    action_input,
                    pddl_action_name,
                    pddl_action_input,
                )

            # reformatting is needed
            retry_count += 1
            if result["parser_flag"] == INVALID_FORMATTING_REDO:
                # just re-prompt without any extra guidance
                continue

            elif (
                result["parser_flag"] == INVALID_FORMATTING_ACTION
                or result["parser_flag"] == INVALID_FORMATTING_ACTION_KWARGS
            ):
                # add reformatting guidance to the temporary history to guide
                # the model
                thought = "Thought: " + result["thought"]
                action = "Action: " + result["action"]
                action_input = "Action Input: " + result["action_input"]
                observation = "Observation: " + result["reformatting_instructions"]
                new_history = ("\n").join([thought, action, action_input, observation])
                input_dict["history"] = orig_history + "\n\n" + new_history
                continue

            raise NotImplementedError()
        if "none" or "n/a" in result["action"].lower():
            return TaskCompletionAction(self.agent_id, result["thought"])
        raise Exception("Max retry limit reached.")

    def get_total_token_count(self) -> int:
        total_token_count = 0
        for r in self.model_response_list:
            total_token_count += r.token_count
        return total_token_count

    def _parse_model_output(self, llm_output: str) -> Dict:
        """Parses the output from a react agent.

        Args:
            llm_output (str): the output from the React agent in string
                format

        Returns:
            (dict): returns a dictionary with the following keys:
                thought (str): the thought provided by the reasoning step of the LLM
                action (str): the action name for the function call
                action_input (str): the action input that the llm predicted. this
                    is a dictionary in string format representing the kwargs for
                    the predicted action
                reformatting_instructions (str): if reformatting is required, here
                    is the string for the reformatting instructions
                parser_flag (int): 0 for valid format and action prediction
                                   1 for valid format and task completion prediction
                                   2 + means invalid formatting:
                                   2 - REDO -> reprompt the model without providing
                                        additional formatting feedback
                                   3 - INVALID ACTION -> reprompt the model to choose
                                        from the correct set of actions
                                   4 - INVALID ACTION KWARGS -> reprompt the model to
                                        format the action arguments appropriately based
                                        on its predicted actions
        """

        # Check if agent should finish
        if "Task Complete:" in llm_output:
            idx = llm_output.find("Task Complete:")
            output = llm_output[idx:]
            self.final_answer = output
            return {
                "thought": output,
                "action": None,
                "action_input": None,
                "reformatting_instructions": None,
                "parser_flag": VALID_TASK_COMPLETION_PRED,
            }

        thought_pattern = r"Thought:\s*(.+)"
        action_pattern = r"Action:\s*(.+)"
        action_input_pattern = r"Action Input:\s*(.+)"

        thought = re.search(thought_pattern, llm_output)
        action = re.search(action_pattern, llm_output)
        action_input = re.search(action_input_pattern, llm_output)

        if action is None or thought is None or action_input is None:
            # for this type of format, no reprompting is needed.
            # simply ask the llm to predict an action again and this
            # type of formatting error is usually resolved
            return {
                "thought": None,
                "action": None,
                "action_input": None,
                "reformatting_instructions": "",
                "parser_flag": INVALID_FORMATTING_REDO,
            }

        thought = thought.group(1).strip().replace("*", "")
        action = action.group(1).strip().replace("*", "")
        action_input = action_input.group(1).replace("*", "")

        # verify the action and action input
        if not self._verify_action(action):
            valid_actions = [a.llm_action_name for a in self.actions]
            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "reformatting_instructions": f"Invalid action: {action}. The predicted action must be one of {valid_actions}",
                "parser_flag": INVALID_FORMATTING_ACTION,
            }
        if not self._verify_action_kwargs(action, action_input):
            action_schema = [i for i in self.actions if i.llm_action_name == action][0]
            kwargs_desc = str(action_schema.llm_arg_desc)
            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "reformatting_instructions": f"The input to {action} must be in the following format: {kwargs_desc}.",
                "parser_flag": INVALID_FORMATTING_ACTION_KWARGS,
            }

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "reformatting_instructions": None,
            "parser_flag": VALID_ACTION_PREDICTION,
        }

    def _verify_action(self, action: str) -> bool:
        """Given an LLM predicted action, make sure that it is
        a valid action.

        Args:
            action (str): the name of the action predicted by the LLM

        Returns:
            (bool): True if the action is valid, otherwise False
        """
        valid_actions = [i.llm_action_name for i in self.actions]
        if action in valid_actions:
            return True
        else:
            return False

    def _verify_action_kwargs(self, action: str, kwargs: str) -> bool:
        """Given an LLM predicted action and action kwargs, make sure
        that the predicted kwargs match the schema of the predicted action.

        Args:
            action (str): the name of the action predicted by the LLM
            kwargs (str): the predicted keyword arguments for the predicted
                action. provided as a dictionary in string format

        Returns:
            (bool): True if the kwargs correspond to the action schema.
                otherwise False
        """
        valid_actions = [i.llm_action_name for i in self.actions]
        if action not in valid_actions:
            raise ValueError(
                f"Action {action} invalid. Must be one of {valid_actions}."
            )

        # attempt to parse the string to a python dictionary
        try:
            kwargs = eval(kwargs)
        except:
            # kwargs formatted incorrectly as we could not parse to Dict
            return False
        if not isinstance(kwargs, Dict):
            return False

        # ensure no duplicate arguments
        kwargs_list = list(kwargs.keys())
        kwargs_set = set(kwargs.keys())
        if len(kwargs_list) != len(kwargs_set):
            return False

        action_schema = [a for a in self.actions if a.llm_action_name == action][0]
        true_kwargs = set(action_schema.llm_arg_desc.keys())

        for kwarg in kwargs:
            if kwarg not in true_kwargs:
                return False

        return True

    def _get_partial_vars(self) -> None:
        """The prompt template contains two types of variables:

        partial variables - variables that are initialized once
            at the beginning of the agent inception and remain
            static thereafter

        input variables - variables that are initialized before
            each call to the llm.

        Here, we will return the template with the partial variables
            added into the template string.

        Args:
            template (str): the initial LLM system prompt

        Returns:
            template_formatted_with_partial_vars (str): the original
                template but with the partial variables formatted
                into the template.
        """

        # create a description for each kwarg to be added onto the
        # base action description
        action_descs = list()
        for action in self.actions:
            action_name = action.llm_action_name
            base_desc = action.llm_action_desc
            kwarg_prefix = " Action input in JSON format: "
            kwargs = str(action.llm_arg_desc)
            full_desc = action_name + ": " + base_desc + kwarg_prefix + kwargs
            action_descs.append(full_desc)

        actions = ("\n").join(action_descs)
        action_names = (", ").join([i.llm_action_name for i in self.actions])

        return {"actions": actions, "action_names": action_names}


DEFAULT_REACT_INPUT_VARS = [
    "task",
    "discovered_objects",
    "history",
    "reformatting_instructions",
]

DEFAULT_REACT_PARTIAL_VARS = ["actions", "action_names"]

DEFAULT_REACT_TEMPLATE = """You are a robot who can use actions to take action in the real world.  You have access to the following actions:

{actions}

Strictly use the following format:

Thought: you should always think about what to do and replan if needed
Action: the action to take, should be one of [{action_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have successfully completed my task
Task Complete: Provide your answer justifying why the task has been completed.

Note: You may need to search for objects inside likely receptacles in the environment.

Begin!

The following objects have been discovered in the environment: <s> [INST] {discovered_objects} [/INST]

Task: <s> [INST] {task} [/INST]

{history}"""

AI2THOR_DEFAULT_REACT_TEMPLATE = """You are a robot who can use actions to take action in the real world.  You have access to the following actions:

{actions}

Strictly use the following format:

Thought: you should always think about what to do and replan if needed
Action: the action to take, should be one of [{action_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have successfully completed my task
Task Complete: Provide your answer justifying why the task has been completed.

Note: You may need to search for objects inside likely receptacles in the environment.

Note: To turn on a StoveBurner, you must toggle on the corresponding StoveKnob.

Begin!

The following objects have been discovered in the environment: <s> [INST] {discovered_objects} [/INST]

Task: <s> [INST] {task} [/INST]

{history}"""

