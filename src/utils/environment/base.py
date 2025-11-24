"""
Maintains a formal Environemnt class that provides a standardized
way for agents to interact with and receive information from an
arbitrary simulation or real world environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from omegaconf import DictConfig

from src.agent.single_agent.base import BaseAgent
from src.tasks.base import TaskMetadata
from src.utils.state.agent_state import AgentState
from src.utils.state.object_database import ObjectDatabase


class Environment(ABC):

    # an environment must contain an object database and
    # a list of agent states for each agent in the environment
    obj_db = ObjectDatabase()
    agent_states: Dict[int, AgentState] = dict()
    agents: Dict[int, BaseAgent] = dict()
    env_steps: int = 0

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, cfg: DictConfig, agent_count: int) -> None:
        """Initializes the environment."""
        pass

    @abstractmethod
    def initialize_agents(self, agents: List[BaseAgent]) -> None:
        """Initializes each agent in the environment."""
        pass

    @abstractmethod
    def reset(self, task_metadata: TaskMetadata) -> None:
        """Resets the environment to the starting state given
        by the task metadata."""
        pass

    @abstractmethod
    def step(self, agent_id: int, **kwargs):
        """Take a step in the environment. Should also handle
        updating the environment state in the object database
        as well as updating the agent states."""
        pass

    @abstractmethod
    def check_if_goal_state_reached(self, task_metadata: TaskMetadata) -> bool:
        """Determine whether or not the goal state for the current
        task has been achieved in the environment."""
        pass

    def get_object_database(self) -> ObjectDatabase:
        return self.obj_db

    def get_agent_state(self, agent_id: int) -> AgentState:
        return self.agent_states[agent_id]

    def get_agent(self, agent_id: int) -> BaseAgent:
        return self.agents[agent_id]
