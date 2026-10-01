import io
import random
from typing import TypedDict

from PIL import Image
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    name: str
    nums: list[int]
    limit: int


def greeting(state: AgentState) -> AgentState:
    """This function greets the user."""
    state["name"] = f"Hi there, {state['name']}!"
    state["nums"] = []
    assert state["limit"] > 0
    return state


def random_number(state: AgentState) -> AgentState:
    """This function generates a random number."""
    state["nums"].append(random.randint(0, 100))
    return state


def should_continue(state: AgentState) -> AgentState:
    """This function checks if the loop should continue."""
    if len(state["nums"]) < state["limit"]:
        return "loop"
    else:
        return "exit"


graph = StateGraph(AgentState)
graph.add_node("greeting", greeting)
graph.add_node("random_num", random_number)
graph.add_edge(START, "greeting")
graph.add_edge("greeting", "random_num")
graph.add_conditional_edges(
    "random_num",
    should_continue,
    {"loop": "random_num", "exit": END}
)

app = graph.compile()

img = Image.open(io.BytesIO(app.get_graph().draw_mermaid_png()))
img.save("loop_graph.png")

result = app.invoke({"name": "Dima", "limit": 5})
print(result)