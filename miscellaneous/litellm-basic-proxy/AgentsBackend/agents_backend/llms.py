from autogen_ext.models.openai import (
    OpenAIChatCompletionClient
)

llm_gemma3_1b = OpenAIChatCompletionClient(
    model="gemma3:1b",
    base_url="http://localhost:4000/",
    api_key="NULL",
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
        "structured_output": False
    }
)