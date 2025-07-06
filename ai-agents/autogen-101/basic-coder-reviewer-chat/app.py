import json
import os
from autogen import AssistantAgent

llm_config = {
    "model": "mistral-small-2503",
    "api_key": os.environ.get("MISTRAL_API_KEY"),
    "api_type": "mistral",
    "base_url": os.environ.get("MISTRAL_BASE_URL"),
}

with open("system_messages.json", "r") as f:
    system_messages = json.load(f)

coding_assistant = AssistantAgent(
    name="coding_assistant", 
    llm_config=llm_config,
    system_message=system_messages["coding_assistant"]
)

reviewer_assistant = AssistantAgent(
    name="reviewer_assistant",
    llm_config=llm_config,
    system_message=system_messages["reviewer_assistant"]
)

reviewer_assistant.initiate_chat(coding_assistant, message="Brad, write me a function function to compute the sum of n integers.", max_turns=2)
