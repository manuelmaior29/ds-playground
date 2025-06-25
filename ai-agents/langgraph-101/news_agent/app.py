from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from langchain_community.llms.ollama import Ollama
from langchain_community.tools import DuckDuckGoSearchRun

from tools import weather_info_tool

# LLM definition
llm = Ollama(model="qwen2.5:14b")  # Check spelling of model name on your local Ollama

# Tool list definition
search_tool = DuckDuckGoSearchRun()
tools = [weather_info_tool]

# Agent state definition
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Assistant function
def assistant(state: AgentState):
    last_message = state["messages"][-1].content
    response = llm.invoke(last_message)
    return {"messages": [HumanMessage(content=response)]}

# Agent graph
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
agent = builder.compile()

# Example usage
messages = [HumanMessage(content="What is the weather in Cluj?")]
response = agent.invoke({"messages": messages})

print("🎩 Agent's Response:")
print(response['messages'][-1].content)
