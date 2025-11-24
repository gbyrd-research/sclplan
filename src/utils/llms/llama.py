"""
Defines LLM classes for the Llama family of Large Language Models.
"""

import ollama

from src.utils.llms.base import ConceptAgentCustomLLM, LLMResponse

class Llama(ConceptAgentCustomLLM):

    def __init__(self, model_name: str, **kwargs):
        self.options = dict(kwargs)
        self.model_name = model_name
        super().__init__()

    def _invoke(
        self, 
        prompt: str = None, 
        messages: list[dict[str]] = None
    ) -> LLMResponse:

        if prompt is None and messages is None:
            raise Exception("A user prompt or sequence of messages must be passed as a parameter.")
        
        if prompt is not None and messages is not None:
            raise Exception("Cannot pass both a user prompt and a user message. Is ambigious.")

        if prompt is not None:
            response = ollama.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                options=self.options
            )
            
            response_str = response.message.content
            token_count = response.eval_count + response.prompt_eval_count
            return LLMResponse(
                content=response_str,
                token_count=token_count,
                full_model_response=response,
            )
        elif messages is not None:
            response = ollama.chat(
                messages=messages,
                model=self.model_name,
                options=self.options
            )
            
            response_str = response.message.content
            token_count = response.eval_count + response.prompt_eval_count
            return LLMResponse(
                content=response_str,
                token_count=token_count,
                full_model_response=response,
            )
        else:
            raise Exception("Should not get here. Something is wrong.")

class Llama_1b(Llama):
    def __init__(self, **kwargs):
        super().__init__(model_name="llama3.2:1b", **kwargs)

class Llama_3b(Llama):
    def __init__(self, **kwargs):
        super().__init__(model_name="llama3.2:3b", **kwargs)

class Llama_8b(Llama):
    def __init__(self, **kwargs):
        super().__init__(model_name="llama3.1", **kwargs)

class Llama_70b(Llama):
    def __init__(self, **kwargs):
        super().__init__(model_name="llama3.3", **kwargs)