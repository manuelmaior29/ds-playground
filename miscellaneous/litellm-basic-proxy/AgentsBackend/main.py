import asyncio

from autogen_agentchat.agents import (
    AssistantAgent
)

from agents_backend.llms import llm_gemma3_1b

async def main():
    agent = AssistantAgent(
        name="dummy_agent",
        system_message="You are a dummy agent. Your name is Dummy.",
        model_client=llm_gemma3_1b,
        tools=[]
    )
    await agent.run(task="Hello!")

if __name__ == "__main__":
    asyncio.run(main())