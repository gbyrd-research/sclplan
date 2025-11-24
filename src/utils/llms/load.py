import os
from typing import Dict, List, Optional

from src.utils.llms.base import ConceptAgentCustomLLM
from src.utils.llms.llama import Llama_1b, Llama_3b, Llama_8b, Llama_70b
from src.utils.llms.openai import GPT3_5, GPT4oMini, GPT4o


def get_llm(llm: str, **llm_kwargs) -> ConceptAgentCustomLLM:
    """Return a ConceptAgentCustomLLM class for standardized LLM
    inference.

    Args:
        llm (str): the name of the LLM chat object to return
        **llm_kwargs (keyword_arguments): any keyword arguments that you wish
            to pass to the LLM during instantiation

    Returns:
        ConceptAgentCustomLLM : repo-specific standardized LLM class
    """
    valid_llm_names = {
        "llama1b", 
        "llama3b", 
        "llama8b", 
        "llama70b",
        "gpt-3.5-turbo",
        "gpt-4o-mini", 
        "gpt-4o"
    }
    
    if llm == "llama1b":
        return Llama_1b(**llm_kwargs)
    elif llm == "llama3b":
        return Llama_3b(**llm_kwargs)
    elif llm == "llama8b":
        return Llama_8b(**llm_kwargs)
    elif llm == "llama70b":
        return Llama_70b(**llm_kwargs)
    elif llm == "gpt-3.5-turbo":
        return GPT3_5(**llm_kwargs)
    elif llm == "gpt-4o-mini":
        return GPT4oMini(**llm_kwargs)
    elif llm == "gpt-4o":
        return GPT4o(**llm_kwargs)
    else:
        raise Exception(f"Model name {llm} not valid. Must be one of {valid_llm_names}")
