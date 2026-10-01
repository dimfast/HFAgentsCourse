from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: list[HumanMessage]


llm = ChatOllama(model="qwen2.5:3b", temperature=0, client_kwargs={"timeout": 60.0})


def process(state: AgentState) -> AgentState:
    """Solve the request using the LLM."""
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = "What is the capital of France?"
result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
print(result)