from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain_community.tools import (
    DuckDuckGoSearchRun
)

from utils import (
    get_question
)
from tools import (
    search_wikipedia,
    get_unreversed_text
)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def build_graph():
    system_prompt = ""
    try:
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("\u26A0️  system_prompt.txt not found. Using default system prompt.")

    def call_model(state: AgentState) -> dict:
        """
        Invokes the LLM with the current messages and returns its response.
        """
        messages = state["messages"]
        sys_msg = SystemMessage(content=system_prompt)
        response = llm_with_tools.invoke([sys_msg] + messages)
        return {"messages": [response]}
    
    tools = [
        search_wikipedia,
        get_unreversed_text,
        DuckDuckGoSearchRun()
    ]
    
    llm = ChatOllama(model="qwen3:8b", verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(state_schema=AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        tools_condition
    )
    graph.add_edge("tools", "call_model")
    agent = graph.compile()
    return agent

class BasicAgent:
    def __init__(self):
        self.graph = build_graph()

    def __call__(self, question: str, verbose: bool = False) -> dict:
        print(f"Agent received question (first 50 chars): {question}...")
        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})
        if verbose:
            for message in messages['messages']:
                message.pretty_print()
        answer = messages['messages'][-1].content
        return answer[14:]

if __name__ == "__main__": 
    tools = [
        search_wikipedia,
        get_unreversed_text,
        DuckDuckGoSearchRun()
    ]
    
    llm = ChatOllama(model="qwen3:8b", verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    try:
        system_prompt = ""
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("\u26A0️  system_prompt.txt not found. Using default system prompt.")
    
    agent = BasicAgent()
    print("Agent started. Type 'exit' to quit.")
    while True:

        user_input = input("================================== You ==================================\n")
        if user_input.lower() == 'exit':
            break
        
        response = agent(question=user_input, verbose=True)