from abc import ABC, abstractmethod

from colorama import Fore, Style


class Tasks(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class TaskMetadata:
    def __init__(self, scene: str, task_name: str, task_natural_language: str):
        """Provides metadata that describes a specific task.

        Args:
            scene (str): The AI2Thor scene in which the task is conducted
            task_name (str): The name of the task
            task_natural_language (str): The task specified in natural language
            pddl_goal_block (List[str]): a list of strings that when written
                to a file with file.writelines(pddl_goal_block) will create
                a pddl :goal block. This can be used to determine whether
                the action was completed successfully (goal state reached)
            agents_overview (List[Dict]): the overview of the multi-agent
                team

        Returns:
            None
        """
        self.scene = scene
        self.task_name = task_name
        self.task_natural_language = task_natural_language

    def print(self):
        print(Fore.YELLOW)
        print("\n")
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
        print(Style.RESET_ALL)

    def get_task_information(self):
        task_info = {
            "scene": self.scene,
            "task_name": self.task_name,
            "task_natural_language": self.task_natural_language
        }
        return task_info
