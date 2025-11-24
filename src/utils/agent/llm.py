import os
from typing import Dict, List, Optional

import ollama
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.runnables.base import Runnable
from langchain_openai import AzureChatOpenAI


def get_llm(llm: str, **llm_kwargs):
    """Return a langchain chat object for various LLMs.

    Args:
        llm (str): the name of the LLM chat object to return
        **llm_kwargs (keyword_arguments): any keyword arguments that you wish
            to pass to the LLM during instantiation

    Returns:
        langchain_runnable : the langchain object that can be used to call
            the LLM
    """
    valid_llm_names = {"llama3", "llama3.1", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}

    # ensure you have the correct key
    api_key = os.getenv("AZURE_OPENAI_KEY")
    assert (
        api_key is not None
    ), "Error: Make sure to set the AZURE_OPENAI_KEY environment variable"
    if llm == "llama3.1_custom":
        return

    if llm == "llama3":
        return ChatOllama(model="llama3", **llm_kwargs)
    elif llm == "llama3.1":
        return ChatOllama(model="llama3.1", **llm_kwargs)
    elif llm == "gpt-4o":
        return AzureChatOpenAI(
            azure_endpoint="https://arlviz.openai.azure.com/",
            api_key=api_key,  # type:ignore
            api_version="2024-02-15-preview",
            azure_deployment="arlviz-gpt4",
            **llm_kwargs,
        )
    elif llm == "gpt-4o-mini":
        return AzureChatOpenAI(
            azure_endpoint="https://arlviz.openai.azure.com/",
            api_key=api_key,  # type:ignore
            api_version="2024-02-15-preview",
            azure_deployment="gpt4o-mini",
            **llm_kwargs,
        )
    elif llm == "gpt-3.5-turbo":
        return AzureChatOpenAI(
            azure_endpoint="https://arlviz.openai.azure.com/",
            api_key=api_key,  # type:ignore
            api_version="2024-02-15-preview",
            azure_deployment="test01",
            **llm_kwargs,
        )
    else:
        raise Exception(f"Model name {llm} not valid. Must be one of {valid_llm_names}")


def get_basic_runnable(
    template: str,
    llm: str,
    input_vars: List[str],
    partial_vars: Dict,
    prompt_template: Optional[PromptTemplate] = None,
    **llm_kwargs,
) -> Runnable:
    """Allows you to get a Langchain runnable that allows you to prompt a
    language model and receive output. This is a basic use of Langchain
    and is not considered an agent. It is essentially just a template that
    you can use to query an LLM.

    Args:
        template (str): the template for the prompt
        llm (str): the name of the LLM model you wish to load
        input_vars (List[str]): a list of input variables that you wish to
            be able to modify whenever you call the runnable. essentially
            your input to the query
        partial_vals (Dict): a dictionary matching the variables in the
            template that you wish to set at the beginning of the prompt
            creation, but will be kept static thereafter

    Return:
        (runnable): a Langchain runnable that you will be able to query
            using runnable.invoke(input_vars)

    """
    llm_model = get_llm(llm, **llm_kwargs)
    if prompt_template is None:
        prompt = PromptTemplate(
            template=template,
            input_variables=input_vars,
            partial_variables=partial_vars,
        )
    else:
        prompt = prompt_template(
            template=template,
            input_variables=input_vars,
            partial_variables=partial_vars,
        )
    return prompt | llm_model
