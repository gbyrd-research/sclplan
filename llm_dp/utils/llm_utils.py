import json
import os
import pickle
import time
import ollama

from openai import OpenAI
from openai._exceptions import RateLimitError
from llm_dp.utils.config import LLMDPConfig
import openai

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint="https://arlviz.openai.azure.com/",
            api_version="2024-02-01"
        )
# client = OpenAI(api_key=LLMDPConfig.openai_api_key)

# we will add the different LLMs for experimentation 
# (llama8b, llama70b, gpt-4o-mini, gpt-4o)
def llm(llm_messages: list[str], stop=None) -> tuple[str, dict[str, int]]:
    model = LLMDPConfig.llm_model
    try:
        if "gpt" in model:
            completion = client.chat.completions.create(
                model=model,
                messages=llm_messages,
                temperature=0.0,
                max_tokens=100,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop,
            )
            with open("llmdp_input.json", "w") as file:
                import json
                llm_messages.append({"output": completion.choices[0].message.content})
                json.dump(llm_messages, file, indent=3)
            return completion.choices[0].message.content, dict(completion.usage)
        elif "llama" in model:
            if model == "llama1b":
                model = "llama3.2:1b"
            elif model == "llama3b":
                model = "llama3.2"
            elif model == "llama8b":
                model = "llama3.1"
            elif model == "llama70b":
                model = "llama3.3"
            else:
                raise NotImplementedError()
            response = ollama.chat(
                messages=llm_messages,
                model=model,
                options={"stop": stop}
            )
            token_info = {
                "completion_tokens": response.eval_count, 
                "prompt_tokens": response.prompt_eval_count,
                "total_tokens": response.eval_count + response.prompt_eval_count
            }
            return response.message.content, token_info

    except RateLimitError:
        time.sleep(10)
        return llm(llm_messages, stop=stop)


def llm_cache(
    llm_messages: list[dict[str]],
    stop=None,
    temperature=0,
) -> tuple[str, dict[str, int]]:
    # If temperature is greater than 0, then skip cache
    if temperature > 0:
        return llm(llm_messages, stop=stop)

    # Cache the openai responses in pickle file
    cache_file = (
        f"{LLMDPConfig.output_dir}/llm_responses_{LLMDPConfig.llm_model}.pickle"
    )
    llm_responses = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            llm_responses = pickle.load(f)

    key = json.dumps(llm_messages)
    if key not in llm_responses:
        print("Not in cache")
        generated_content, token_usage = llm(llm_messages, stop=stop)
        llm_responses[key] = (generated_content, token_usage)
        with open(cache_file, "wb") as f:
            pickle.dump(llm_responses, f)

    return llm_responses[key]
