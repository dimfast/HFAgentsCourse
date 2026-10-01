from typing import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    age: int
    final: str


def first_node(state: AgentState) -> AgentState:
    """This is the first node in the graph."""
    state["final"] = f"Hi {state['name']}!"
    return state


def second_node(state: AgentState) -> AgentState:
    """This is the second node in the graph."""
    state["final"] += f" You are {state['age']} years old."
    return state


graph = StateGraph(AgentState)
graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_edge("first_node", "second_node")
graph.set_entry_point("first_node")
graph.set_finish_point("second_node")

app = graph.compile()

result = app.invoke({"name": "Dima", "age": 33})
print(result)