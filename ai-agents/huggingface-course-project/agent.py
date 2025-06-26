from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama

from tools import search_wikipedia_tool, weather_tool

tools = [search_wikipedia_tool, weather_tool]
llm = ChatOllama(model="llama3-groq-tool-use:8b", verbose=True)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    
# Nodes
def call_model(state: AgentState) -> dict:
    """
    Invokes the LLM with the current messages and returns its response.
    """
    messages = state["messages"]
    # print(f"\n--- LLM Input Messages ---")
    # for msg in messages:
    #     print(f"  {type(msg).__name__}: {msg.content[:100]}...")
    #     if isinstance(msg, AIMessage) and msg.tool_calls:
    #         print(f"    (Tool Calls in history: {msg.tool_calls})")
    response = llm_with_tools.invoke(messages)
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
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        print("Agent:")
        response = agent.invoke(input=inputs)
        print(response['messages'][-1].content)
