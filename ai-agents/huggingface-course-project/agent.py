from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain_community.tools import (
    DuckDuckGoSearchRun
)

from tools import (
    search_wikipedia_tool,
    text_unreverser_tool
)

tools = [
    search_wikipedia_tool,
    text_unreverser_tool,
    DuckDuckGoSearchRun()
]
llm = ChatOllama(model="qwen3:8b", verbose=True)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    
# Nodes
def call_model(state: AgentState) -> dict:
    """
    Invokes the LLM with the current messages and returns its response.
    """
    messages = state["messages"]
    sys_msg = SystemMessage(content=f"""You are a helpful assistant that answers to questions directly.
                            Use any tool available to you to answer the question, but only if it makes sense. 
                            If you cannot answer the question, answer with "I don't know".""")
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": [response]}

# Graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("call_model", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("call_model")
workflow.add_conditional_edges(
    "call_model",
    tools_condition
)
workflow.add_edge("tools", "call_model")
agent = workflow.compile()

if __name__ == "__main__":
    print("Agent started. Type 'exit' to quit.")
    while True:

        user_input = input("================================== You ==================================\n")
        if user_input.lower() == 'exit':
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        response = agent.invoke(input=inputs)
        # Print the steps
        for message in response["messages"]:
            if not isinstance(message, HumanMessage):
                message.pretty_print()
