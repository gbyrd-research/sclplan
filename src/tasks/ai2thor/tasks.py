from ai2thor.controller import Controller

from src.tasks.ai2thor.goal_states import *
from src.tasks.ai2thor.team_configs import *
from src.tasks.ai2thor.verification import *
from src.tasks.base import TaskMetadata, Tasks

cur_file_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(cur_file_path)

# TODO: Save out goal states and read from file instead of initializing the
# AI2Thor controller each time


class AI2ThorTaskMetadata(TaskMetadata):
    def __init__(
        self,
        scene: str,
        task_name: str,
        task_natural_language: str,
        team_cfg: List[Dict],
        verification_func: callable,
    ):
        super().__init__(scene, task_name, task_natural_language)
        self.team_cfg = team_cfg
        self.verification_func = verification_func


class AI2ThorTasks(Tasks):
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.tasks = self._get_tasks(task_type)

    def _get_tasks(self, task_type: str) -> List[TaskMetadata]:
        """Given the task type return a list of TaskMetadatas."""
        valid_task_types = ["single_agent", "multi_agent", "single_agent_hard"]
        if task_type == "single_agent":
            return generate_single_agent_tasks()
        elif task_type == "multi_agent":
            return generate_multi_agent_tasks()
        elif task_type == "single_agent_hard":
            return generate_single_agent_hard_tasks()
        else:
            raise NotImplementedError(
                f"Invalid task type: {task_type}. Must be one of {valid_task_types}."
            )

    def __iter__(self):  # type:ignore
        self.index = 0
        return self

    def __next__(self):  # type:ignore
        if self.index < len(self.tasks):
            value = self.tasks[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration


def generate_single_agent_tasks() -> List[TaskMetadata]:
    single_agent_tasks: List[TaskMetadata] = list()
    single_agent_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Test_Task_0000",
            "Place a kettle on the stoveburner, place the apple in the fridge, and slice the bread.",
            team_cfg_single_agent,
            lambda x: "Need to add verification function.",
        )
    )

    single_agent_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Test_Task_0001",
            "Cook an egg.",
            team_cfg_single_agent,
            lambda x: "Need to add verification function.",
        )
    )

    single_agent_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Test_Task_0002",
            "Put an egg in the bowl, place a kettle on the stove burner, slice the bread.",
            team_cfg_multi_agent_1,
            lambda x: "Need to add verification function.",
        )
    )

    return single_agent_tasks


def generate_single_agent_hard_tasks() -> list[AI2ThorTaskMetadata]:
    single_agent_hard_tasks: list[AI2ThorTaskMetadata] = list()

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_000",
            "Cook an egg.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if the egg was placed in a container and then heated up in some way.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_001",
            "Rinse off the pan.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if the pan was placed in the sink and the faucet was toggled on while the pan was still in the sink.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_002",
            "Heat up the bread. Do not use the stove burner.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if the bread was either toasted or heated up in the microwave.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_003",
            "Slice the bread and toast one slice. Place the slice in the pan.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if a slice of bread was toasted and then places in the pan.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_004",
            "Chill the tomato.",
            team_cfg_single_agent,
            hard_task_004_verification,
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_005",
            "Place an apple on the counter and slice it, rinse one slice, then place the slice in a bowl.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if an apple was sliced, a slice was placed in the sink while the faucet is turned on, and that same slice was then placed in a bowl.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan5",
            "Single_Agent_Hard_Task_006",
            "Take the tomato off the plate and slice it on the counter. Next, slice the bread. Put one tomato slice and two slices of bread on the plate.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if the tomato was sliced before the bread and if there is one tomato slice and two slices of bread on a plate.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_007",
            "Using a microwave, cook an egg in a bowl and place the bowl on a counter.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if the egg was placed in a bowl, then placed in the microwave, and the microwave was toggled on. Then the bowl was placed on the counter with the egg in it.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_008",
            "Clear out the fridge, placing everything inside it on the counter",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if there is nothing left inside the fridge and everything inside the fridge was placed on the counter.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_009",
            "Slice the tomato, chill a slice, then put the chilled slice in a bowl.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if a chilled slice of tomato is in a bowl at the end of the task.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan5",
            "Single_Agent_Hard_Task_010",
            "Rinse a bowl and then place it in the microwave.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if a bowl was in the sink while the faucet was toggled on and was then placed in the microwave.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_011",
            "Use a bowl to cook a potato in the microwave and then place the cooked potato on a plate.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if the potato was placed on a plate and then placed in the microwave, the microwave was then toggled on, and the potato was then removed from the bowl and placed on a plate.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_012",
            "I have put my K-cup in the coffee machine. Now get a mug for me and make me coffee.",
            team_cfg_single_agent,
            hard_task_012_verification,
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_013",
            "Cook a potato.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if a potato is heated up in any way.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_014",
            "Rinse off a fork and put it in a bowl with a slice of tomato.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if a fork is placed in the sink while the faucet is on before being placed in a bowl with a slice of tomato.",
        )
    )

    single_agent_hard_tasks.append(
        AI2ThorTaskMetadata(
            "FloorPlan1",
            "Single_Agent_Hard_Task_015",
            "Boil a pot of water.",
            team_cfg_single_agent,
            lambda x: "Requires Manual Verification: True if a pot is filled with water and is then heated up.",
        )
    )

    # the below tasks have bugs so they are excluded

    # single_agent_hard_tasks.append(
    #     AI2ThorTaskMetadata(
    #         "FloorPlan10",
    #         "Single_Agent_Hard_Task_016",
    #         "Wash some lettuce and slice the lettuce on the counter before placing two lettuce slices in a bowl.",
    #         team_cfg_single_agent,
    #         lambda x: "Requires Manual Verification: True if the lettuce is placed in the sink while the faucet is toggled on, then the lettuce is then moved to the counter, sliced, and two of the slices are placed in a bowl.",
    #     )
    # )

    # single_agent_hard_tasks.append(
    #     AI2ThorTaskMetadata(
    #         "FloorPlan10",
    #         "Single_Agent_Hard_Task_017",
    #         "Place a slice of lettuce and a slice of tomato in the pan.",
    #         team_cfg_single_agent,
    #         lambda x: "Requires Manual Verification: True if there is a slice of lettuce and a slice of tomato in the pan.",
    #     )
    # )

    # single_agent_hard_tasks.append(
    #     AI2ThorTaskMetadata(
    #         "FloorPlan15",
    #         "Single_Agent_Hard_Task_018",
    #         "Heat up a tomato in the microwave, but first place the tomato in a bowl.",
    #         team_cfg_single_agent,
    #         lambda x: "Requires Manual Verification: True if a tomato is placed in a bowl, then placed in the microwave, before the microwave is toggled on.",
    #     )
    # )

    # single_agent_hard_tasks.append(
    #     AI2ThorTaskMetadata(
    #         "FloorPlan15",
    #         "Single_Agent_Hard_Task_018",
    #         "Chill an apple, then slice it.",
    #         team_cfg_single_agent,
    #         lambda x: "Requires Manual Verification: True if an apple is placed in the fridge and is THEN sliced. The order matters here.",
    #     )
    # )

    # single_agent_hard_tasks.append(
    #     AI2ThorTaskMetadata(
    #         "FloorPlan15",
    #         "Single_Agent_Hard_Task_019",
    #         "Using the coffee machine, make me some coffee in the mug. Then, chill the coffee.",
    #         team_cfg_single_agent,
    #         lambda x: "Requires Manual Verification: True if the mug is filled with coffee before it is placed in the fridge.",
    #     )
    # )

    return single_agent_hard_tasks


def generate_tapa_validation_tasks() -> list[AI2ThorTaskMetadata]:
    tapa_val_tasks: list[AI2ThorTaskMetadata] = list()

    # read in the dataset information from the tapa validation json file
    with open(os.path.join(CUR_DIR, "tapa_validation_set.json"), "r") as file:
        tapa_val_set = json.load(file)

    for idx, task_info in enumerate(tapa_val_set):
        tapa_val_tasks.append(
            AI2ThorTaskMetadata(
                task_info["scene_name"],
                f"TaPA_Val_Task_{str(idx).zfill(3)}",
                task_info["instruction"],
                team_cfg_single_agent,
                lambda x: "Requires Manual Verification",
            )
        )

    return tapa_val_tasks


def generate_multi_agent_tasks() -> List[TaskMetadata]:
    raise NotImplementedError("Has not been implemented yet.")
    c = Controller()
    multi_agent_tasks: List[TaskMetadata] = list()
    multi_agent_tasks.append(
        TaskMetadata(
            "FloorPlan1",
            "Test_Task_0000",
            "Cook an egg.",
            task_0000_goal_state("FloorPlan1", c),
            team_cfg_multi_agent_1,
        )
    )

    multi_agent_tasks.append(
        TaskMetadata(
            "FloorPlan1",
            "Test_Task_0001",
            "Place a kettle on the stove, place the apple in the fridge, and slice the bread.",
            task_0001_goal_state("FloorPlan1", c),
            team_cfg_multi_agent_1,
        )
    )

    multi_agent_tasks.append(
        TaskMetadata(
            "FloorPlan1",
            "Test_Task_0002",
            "Place a kettle on the stove, slice the bread, take an egg out of the fridge and put it in the microwave.",
            task_0002_goal_state("FloorPlan1", c),
            team_cfg_multi_agent_1,
        )
    )
    c.stop()
    return multi_agent_tasks
