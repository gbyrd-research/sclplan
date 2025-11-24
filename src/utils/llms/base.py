"""
Defines the custom LLM class interface.
"""

from abc import ABC, abstractmethod
import signal
from typing import Any, List, Optional


class LLMResponse:
    """Defines a standardized data structure to return
    an LLM reponse."""

    def __init__(self, content: str, token_count: int, full_model_response: Any):
        """
        Args:
            content (str): the complete, unfiltered response of the model
                in string format
            token_count (int): the number of tokens used by the model
                (sum of input and output tokens)
            full_model_response (Any): the unprocessed, full output of
                the model (in case you want more information after an
                experiment that you did not account for ahead of time)

        """
        self.content = content
        self.token_count = token_count
        self.full_model_response = full_model_response


class ConceptAgentCustomLLM(ABC):

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.response_history: list[LLMResponse] = list()

    @abstractmethod
    def _invoke(
        self, 
        prompt: str = None, 
        messages: list[dict[str]] = None
    ) -> LLMResponse:
        pass

    def invoke(
        self,
        prompt: str = None,
        messages: list[dict[str]] = None
    ) -> LLMResponse:
        response = self._invoke(prompt, messages)
        self.response_history.append(response)
        return response

