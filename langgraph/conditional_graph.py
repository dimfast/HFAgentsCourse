from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    num1: int
    num2: int
    operation: str
    result: int


def add_numbers(state: AgentState) -> AgentState:
    """This function adds two numbers together."""
    state["result"] = state["num1"] + state["num2"]
    return state


def multiply_numbers(state: AgentState) -> AgentState:
    """This function multiplies two numbers."""
    state["result"] = state["num1"] * state["num2"]
    return state


def decide_operation(state: AgentState) -> AgentState:
    """This function decides which operation to perform."""
    if state["operation"] == "+":
        return "addition_operation"
    elif state["operation"] == "*":
        return "multiplication_operation"
    else:
        raise ValueError(f"Invalid operation: {state['operation']}!")


graph = StateGraph(AgentState)
graph.add_node("add_nums", add_numbers)
graph.add_node("multiply_nums", multiply_numbers)
graph.add_node("router", lambda state: state)
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    decide_operation,
    {
        "addition_operation": "add_nums",
        "multiplication_operation": "multiply_nums",
    }
)
graph.add_edge("add_nums", END)
graph.add_edge("multiply_nums", END)

app = graph.compile()

result = app.invoke({"num1": 5, "num2": 3, "operation": "-"})
print(result["result"])