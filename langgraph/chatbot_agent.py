from typing import TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: list[HumanMessage | AIMessage]


llm = ChatOllama(model="qwen2.5:3b", temperature=0, client_kwargs={"timeout": 60.0})


def process(state: AgentState) -> AgentState:
    """Solve the request using the LLM."""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    # print(result["messages"])
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logging.txt", "w") as f:
    f.write("Your conversation history:\n")

    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            f.write(f"User: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            f.write(f"AI: {msg.content}\n\n")
    f.write("End of conversation.")

print("Conversation logging complete.")