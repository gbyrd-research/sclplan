from pydantic_settings import BaseSettings
from typing import Literal


class LLMDPBaseConfig(BaseSettings):
    openai_api_key: str
    alfworld_data_path: str = "/home/local_byrd/Desktop/concept-agent/src/utils/environment/alfworld/alfworld/alfworld/data"
    alfworld_config_path: str = "/home/local_byrd/Desktop/concept-agent/src/utils/environment/alfworld/alfworld/configs/base_config.yaml"
    pddl_dir: str = "llm_dp/pddl"
    pddl_domain_file: str = "alfworld_domain.pddl"
    planner_solver: Literal["bfs_f", "ff"] = "bfs_f"
    planner_timeout: int = 30
    planner_cpu_count: int = 4
    top_n: int = 3
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    output_dir: str = "output"
    seed: int = 42
    name: str = "llmdp"
    llm_model: str = "gpt-4o-mini"
    sample: Literal["llm", "random"] = (
        "llm"  # whether to use LLM to instantiate beliefs
    )
    use_react_chat: bool = False  # activate ReAct baseline
    random_fallback: bool = True  # activate random fallback
    temperature: float = 0.0  # temperature for LLM sampling

    class Config:
        env_file = ".env"


LLMDPConfig = LLMDPBaseConfig()
