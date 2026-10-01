from typing import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    values: list[int]
    name: str
    result: str


def process_values(state: AgentState) -> AgentState:
    """This function handles multiple different inputs."""
    print(state)
    state["result"] = f"Hey, {state['name']}! Your sum is: {sum(state['values'])}"
    print(state)
    return state


graph = StateGraph(AgentState)
graph.add_node("process_values", process_values)
graph.set_entry_point("process_values")
graph.set_finish_point("process_values")

app = graph.compile()

result = app.invoke({"values": [1, 2, 3, 4], "name": "Dima"})
print(result["result"])