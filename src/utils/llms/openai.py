"""
Defines LLM classes for the OpenAI family of Large Language Models.
"""

import os

import openai

from src.utils.llms.base import ConceptAgentCustomLLM, LLMResponse

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class GPTBase(ConceptAgentCustomLLM):

    def __init__(self, **llm_kwargs):
        # self.azure_openai_endpoint = "https://arlviz.openai.azure.com/"  # Replace with your endpoint
        # openai.api_type = "azure"
        # openai.api_base = self.azure_openai_endpoint
        # openai.api_key = AZURE_OPENAI_API_KEY
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        # openai.api_version = "2024-02-01"
        # self.client = openai.OpenAI()
        self.client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint="https://arlviz.openai.azure.com/",
            api_version="2024-02-01"
        )
        self.model_name = None
        if "top_k" in llm_kwargs:
            del llm_kwargs["top_k"]
        self.options = dict(llm_kwargs)
        super().__init__()


    def _invoke(
        self, 
        prompt: str = None, 
        messages: list[dict[str]] = None,
    ) -> LLMResponse:
        if prompt is None and messages is None:
            raise Exception("A user prompt or sequence of messages must be passed as a parameter.")
        
        if prompt is not None and messages is not None:
            raise Exception("Cannot pass both a user prompt and a user message. Is ambigious.")

        if prompt is not None:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                **self.options
            )
            
            response_str = response.choices[0].message.content
            token_count = response.usage.total_tokens
            return LLMResponse(
                content=response_str,
                token_count=token_count,
                full_model_response=response,
            )
        elif messages is not None:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                **self.options
            )
            # with open("cala_input.json", "w") as file:
            #     import json
            #     messages.append({"output": response.choices[0].message.content})
            #     json.dump(messages, file, indent=3)
            
            response_str = response.choices[0].message.content
            token_count = response.usage.total_tokens
            return LLMResponse(
                content=response_str,
                token_count=token_count,
                full_model_response=response,
            )
        else:
            raise Exception("Should not get here. Something is wrong.")
    
class GPT3_5(GPTBase):
    def __init__(self, **llm_kwargs):
        super().__init__(**llm_kwargs)
        self.model_name = "gpt-3.5-turbo"

class GPT4oMini(GPTBase):
    def __init__(self, **llm_kwargs):
        super().__init__(**llm_kwargs)
        self.model_name = "gpt4o-mini"

class GPT4o(GPTBase):
    def __init__(self, **llm_kwargs):
        super().__init__(**llm_kwargs)
        self.model_name = "gpt-4o"
