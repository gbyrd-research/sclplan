import os
import logging

from hydra.core.hydra_config import HydraConfig


def get_hydra_run_dir() -> str:
    """Returns the path to the hydra run output directory."""
    # Ensure HydraConfig is initialized
    if HydraConfig.initialized():
        output_dir = HydraConfig.get().run.dir
        return output_dir
    else:
        raise RuntimeError(
            "hydra has not yet been initialized. Make sure you are running with hydra."
        )


def ensure_dir_exists(path: str) -> None:
    """Takes in a file path to a directory and creates the directory if
    it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def set_logging_thresholds():
    # set logging thresholds so that they do not flood the terminal
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("ai2thor.build").setLevel(logging.WARNING)
    logging.getLogger("HYDRA").setLevel(logging.WARNING)
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
