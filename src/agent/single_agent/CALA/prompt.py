from typing import Dict

from typing import Dict

GOAL_STATE_TEMPLATE_AI2THOR = """You are a helpful agent that takes in natural language instructions
and decomposes those instructions into goal states for use with various planning
algorithms.

Here is the available predicate information from the PDDL domain file. Please only use these predicates and do not create your own.:
{domain_predicate_info}

Given a task and a description of the environment, you should decompose the task into 
a valid pddl goal state using the above domain information as a corpus of possible predicates.

Strictly adhere to the format in the following examples:
{goal_generation_examples}

Begin!

Task: <s> [INST] {input_task} [/INST]
Discovered Objects: <s> [INST] {discovered_objects} [/INST]"""

GOAL_STATE_TEMPLATE_ALFWORLD = """You are a helpful agent that takes in natural language instructions
and decomposes those instructions into goal states for use with various planning
algorithms.

Here is the available predicate information from the PDDL domain file. Please only use these predicates and do not create your own.:
{domain_predicate_info}

Given a task and a description of the environment, you should decompose the task into 
a valid pddl goal state using the above domain information as a corpus of possible predicates.

Strictly adhere to the format in the following examples:
{goal_generation_examples}

Begin!

Task: <s> [INST] {input_task} [/INST]
"""

GENERATE_GOAL_PROMPT_LLMDP = [
    {
        "role": "system",
        "content": """(define (domain alfred)
(:predicates
(isReceptacle ?o - object) ; true if the object is a receptacle
(atReceptacleLocation ?r - object) ; true if the robot is at the receptacle location
(inReceptacle ?o - object ?r - object) ; true if object ?o is in receptacle ?r
(openable ?r - object) ; true if a receptacle is openable
(opened ?r - object) ; true if a receptacle is opened
(isLight ?o - object) ; true if an object is light source
(examined ?o - object ?l - object) ; whether the object has been looked at with light
(holds ?o - object) ; object ?o is held by robot
(isClean ?o - object) ; true if the object has been cleaned in sink
(isHot ?o - object) ; true if the object has been heated up
(isCool ?o - object) ; true if the object has been cooled
(isSink ?o - object) ; true if the object is a sink
(isMicrowave ?o - object) ; true if the object is a microwave
(isFridge ?o - object) ; true if the object is a fridge
))""",
    },
    {
        "role": "user",
        "content": "Your task is to: put a clean plate in microwave.",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - plate ?r - microwave)
(and (inReceptacle ?t ?r)
(isClean ?t)
)))""",
    },
    {
        "role": "user",
        "content": "Your task is to: examine an alarmclock with the desklamp",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - alarmclock ?l - desklamp)
(and (examined ?t ?l) (holds ?t)
)))""",
    },
    {"role": "user", "content": "Your task is to: put two cellphone in bed"},
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t1 - cellphone ?t2 - cellphone ?r - bed)
(and (inReceptacle ?t1 ?r)
(inReceptacle ?t2 ?r)
(not (= ?t1 ?t2))
)))""",
    },
]

# GOAL_STATE_TEMPLATE = """You are a helpful agent that takes in natural language instructions
# and decomposes those instructions into goal states for use with various planning
# algorithms.

# You have access to several high level, formalized statements to help you generate goal states.
# A list of these statements along with their descriptions are provided below:

# {goal_state_helper_statements}

# Here is a list of attributes that can be attributed to each object:
# {object_attributes_list}

# Given a task and a description of the environment, you should decompose the task into 
# a goal state through the use of the helper statements.

# Strictly use the following format:

# Task: the task you must complete
# Discovered Objects: A list of object ids that have been discovered in the environment.
# Goal State: a python list of goal statements

# Done!

# EXAMPLE 1:

# Task: You should place a fork in the drawer, an apple in the fridge, and turn on the faucet.
# Discovered Objects: Fork_1, Bottle_1, Drawer_1, Drawer_2, Cabinet_1, Fridge_1, Vase_1, Faucet_1, Apple_1, Fork_2, Knife_1, Drawer_3
# Goal State: [object_id_in_object_type(Fork_1, drawer), object_type_in_object_type(Apple_1, fridge), object_id_has_attribute(Faucet_1, isToggled, True)]

# Done!

# EXAMPLE 2:
# Task: You should heat up some food.
# Discovered Objects: Fork_1, Plate_1, Drawer_1, Microwave_1, Egg_1, Apple_1, Knife_1, Fridge_1, Fauce_1, Vase_1, WineBottle_1, Shelf_1
# Goal State: [object_id_in_object_type(Egg_1, plate), object_id_has_attribute(Egg_1, isCooked, True)]

# Done!

# EXAMPLE 3:
# Task: You should close the microwave.
# Discovered Objects: Spoon_1, TrashCan_1, Cabinet_1, Sink_1, LightSwitch_1, Cabinet_2, Spoon_2, Microwave_1
# Goal State: [object_id_has_attribute(Microwave_1, isOpen, False)]

# Done!

# Begin!

# Task: <s> [INST] {input_task} [/INST]
# Discovered Objects: <s> [INST] {discovered_objects} [/INST]"""
