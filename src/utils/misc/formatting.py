"""
This file holds various functions relating to the formatting
of LLM responses, tool calls, etc.
"""

import ast
import inspect
from collections import Counter
from typing import Callable, Dict, List, Tuple


def format_pddl_goal_state(states: List[str]) -> List[str]:
    """Takes in a list of goal states in string form and
    processes these states into a full list of strings that
    when written to a .pddl file with file.writelines() will create
    a valid pddl :goal block."""
    goal_block = list()
    goal_block.append(f"(:goal (and \n")
    goal_block.extend(["\t" + x for x in states])
    goal_block.append("))\n")
    return goal_block


def process_func_in_str_format(func_str: str) -> Tuple[str, List[str]]:
    """Takes in a function in string format, i.e.:
        function_name(arg_1, ..., arg_N)

    And returns a tuple providing the function name and a List of the
    function arguments.

    Args:
        func_str (str): the function call represented in string format

    Return:
        func_name (str): the name of the function
        args (List[str]): a list of the function arguments
    """
    assert func_str.count("(") == 1
    assert func_str.count(")") == 1
    open_paren_idx = func_str.find("(")
    close_paren_idx = func_str.find(")")
    func_name = func_str[:open_paren_idx]
    args = func_str[open_paren_idx + 1 : close_paren_idx].split(",")
    return func_name, args


def format_lowercase_ai2thor_id(ai2thor_id_lcase: str, objects) -> str:
    """Takes in a lowercase version of an ai2thor object id and converts
    it to a corrected id with uppercase letters in the correct location.

    Args:
        ai2thor_id_lcase (str): the ai2thor id in complete lowercase
            form
        objects (ObjectDatabase): the ai2thor tool engine objects database

    Returns:
        ai2thor_id_formatted (str): the ai2thor id in a corrected format
            with uppercase letters where needed
    """
    for ai2thor_id_formatted in objects.env_ids:
        if ai2thor_id_lcase == ai2thor_id_formatted.lower():
            return ai2thor_id_formatted
    raise ValueError(f"No match for target object id: {ai2thor_id_lcase}.")


def get_func_args_desc(func: Callable) -> str:
    """Given a function, provide a string representing a description
    of the arguments of that function provided by the docstring of
    that function.

    Args:
        func (Callable): the function whose parameters you wish to
            get a description of

    Returns:
        arg_desc (str): a string that provides a description of
            the provided function's arguments taken directly from
            the function's docstring
    """
    docstring = func.__doc__

    if docstring is None:
        e_msg = f"You must create docstring for func {func.__name__}."
        raise Exception(e_msg)

    # get the docstring starting at "Args:"
    start_idx = docstring.find("Args:")
    end_idx = docstring.find("Returns:")
    if start_idx != -1:
        if end_idx != -1:
            arg_desc = docstring[start_idx:end_idx]
        else:
            e_msg = f"{func.__name__} docstring must contain 'Returns:'"
            raise Exception(e_msg)
    else:
        e_msg = f"{func.__name__} docstring must contain 'Args:'"
        raise Exception(e_msg)
    return arg_desc


def format_string(input_str: str) -> str:
    """Do basic formatting in the llm produced input string."""
    # remove extraneous non alpha numeric characters
    valid_non_alpha_num_characters = {":", "'", '"', "_", "{", "}", ",", " "}
    formatted_str = ""
    for char in input_str:
        if char.isalpha() or char.isnumeric() or char in valid_non_alpha_num_characters:
            formatted_str += char
    # remove everything outside of the first dictionary
    idx = start_idx = input_str.find("{")
    if idx == -1:
        return formatted_str  # no dictionary in formatted string, so return as is
    idx += 1
    end_idx = None
    left_paren = 1
    right_paren = 0
    while idx < len(input_str):
        if right_paren == left_paren:
            # parenthesis matched, first dictionary closed, so break
            end_idx = idx
            break
        if input_str[idx] == "{":
            left_paren += 1
        elif input_str[idx] == "}":
            right_paren += 1
        idx += 1
    if end_idx is None:
        # the dictionary was not closed. this is a formatting error that
        # will be taken care of when the formatted string is parsed into
        # a dictionary later
        return formatted_str
    # return only the part of the string representing the first fully closed
    # dictionary
    return formatted_str[start_idx:end_idx]


def add_tool_return_prefix(verification_flag: int, initial_obs: str) -> str:
    """Adds a prefix to the tool return observation specifying tool
    success or failure in a standard way."""
    prefix = "Tool Failed: "
    if verification_flag == 0:
        prefix = "Tool Completed Successfully: "
    return prefix + initial_obs


def get_dict_from_str(input_str: str) -> Dict:
    """Takes in an input string from the LLM an attempts to parse
    the string into a dictionary. Expects the string to be loosely
    in the form "{'key', value}".
    """
    # TODO: Maybe there is some more advanced preprocessing we can do here?
    try:
        output_dict = eval(input_str)
        lowercase_dict = dict()
        for key, value in output_dict.items():
            lowercase_dict[key.lower()] = value
        # i have seen some instances of None being added as a key in the dict,
        # so I will account for this here - GB
        if None in lowercase_dict.keys():
            del lowercase_dict[None]
    except:
        return {}  # if cannot parse to dict, return empty dict
    return lowercase_dict


def find_duplicate_keys(dict_str):
    """Searches through a python dictionary represented as a string and
    determines if there are any duplicate keys in the string."""
    try:
        # Parse the string as an abstract syntax tree (AST) dictionary
        tree = ast.parse(dict_str, mode="eval")
        # Extract key-value pairs
        if isinstance(tree.body, ast.Dict):
            keys = [ast.literal_eval(key) for key in tree.body.keys]
            # Count occurrences of each key and find duplicates
            key_counts = Counter(keys)
            duplicates = {key: count for key, count in key_counts.items() if count > 1}
            return duplicates
        else:
            raise ValueError("Input string is not a dictionary.")
    except SyntaxError:
        raise ValueError("Input string is not valid Python dictionary syntax.")
