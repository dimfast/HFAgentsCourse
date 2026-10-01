from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

######### SOME TERMS #########
# Reducer functions - Tells how to merge new data into the current state.
# BaseMessage - The basic class for all message types in LangGraph.
# ToolMessage - Passes data back as content to LLM after it calls a tool.
# SystemMessage - Message for providing instructions to the LLM.


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # add_messages - reducer function for merging new messages into the current state


@tool
def add(a: int, b: int) -> int:
    """This is an addition tool that takes two numbers and returns their sum."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """This is an multiplication tool that takes two numbers and returns their product."""
    return a * b


tools = [add, multiply]

llm = ChatOllama(model="qwen2.5:3b", temperature=0, client_kwargs={"timeout": 60.0}).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability. Please use detailed and short answers."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {"continue": "tools", "end": END}
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream: str):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# inputs = {"messages": [("user", "Add 3 + 4.")]}
# inputs = {"messages": [("user", "Add 3 + 4. Also add 34 + 25.")]}
# inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6.")]}
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke associated with the onbtained calculation result.")]}

print_stream(app.stream(inputs, stream_mode="values"))