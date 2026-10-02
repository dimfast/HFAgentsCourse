from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from final_agent.nodes import AgentState, make_nodes, finalize_answer
from final_agent.questions import get_random_question
from final_agent.tools import web_search, fetch_webpage, read_text_file, python_exec


def route_by_task_type(state):
    task_type = state["task_type"]

    if task_type in {"web", "unknown"}:
        return "web"

    if task_type == "python_file":
        return "python"

    if task_type in {"calculation", "logic", "simple"}:
        return "reasoning"

    return "unsupported"


def should_continue_web(state):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    content = getattr(last_message, "content", "").strip()

    if not content or content.startswith("{") and '"thought"' in content:
        return "retry"

    return "finalize"


def route_after_tools(state):
    active_node = state.get("active_task_node")

    if active_node in {"web_task", "python_task", "reasoning_task"}:
        return active_node

    return "web_task"


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    client_kwargs={"timeout": 60.0},
)

tools = [web_search, fetch_webpage, read_text_file, python_exec]
nodes = make_nodes(llm, tools)

graph = StateGraph(AgentState)

graph.add_node("classify_task", nodes["classify_task"])
graph.add_node("prepare_file", nodes["prepare_file"])
graph.add_node("web_task", nodes["web_task"])
graph.add_node("python_task", nodes["python_task"])
graph.add_node("reasoning_task", nodes["reasoning_task"])
graph.add_node("tools", ToolNode(tools))
graph.add_node("finalize_answer", finalize_answer)

graph.add_edge(START, "classify_task")
graph.add_edge("classify_task", "prepare_file")

graph.add_conditional_edges(
    "prepare_file",
    route_by_task_type,
    {
        "web": "web_task",
        "python": "python_task",
        "reasoning": "reasoning_task",
        "unsupported": END,
    },
)

graph.add_conditional_edges(
    "web_task",
    should_continue_web,
    {"tools": "tools", "retry": "web_task", "finalize": "finalize_answer"},
)

graph.add_conditional_edges(
    "python_task",
    should_continue_web,
    {"tools": "tools", "retry": "python_task", "finalize": "finalize_answer"},
)

graph.add_conditional_edges(
    "reasoning_task",
    should_continue_web,
    {"tools": "tools", "retry": "reasoning_task", "finalize": "finalize_answer"},
)
graph.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "web_task": "web_task",
        "python_task": "python_task",
        "reasoning_task": "reasoning_task",
    },
)

app = graph.compile()

question = get_random_question()

initial_state = {
    "task_id": question["task_id"],
    "question": question["question"],
    "file_name": question.get("file_name", ""),
    "file_path": None,
    "active_task_node": "",
    "task_type": "",
    "answer_format": "",
    "final_answer": "",
    "messages": [],
}

result = app.invoke(initial_state)

print("=" * 80)
print("TASK ID:", question["task_id"])
print("QUESTION:")
print(question["question"])
print("FILE:", question.get("file_name") or "None")
print("=" * 80)

print("CLASSIFIED AS:", result["task_type"])
print("=" * 80)

messages = result.get("messages", [])

if not messages:
    print("NO AGENT MESSAGES")
else:
    for index, message in enumerate(messages, start=1):
        print(f"MESSAGE {index}: {type(message).__name__}")

        content = getattr(message, "content", "")
        if content:
            print("CONTENT:")
            print(content)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("TOOL CALLS:")
            print(tool_calls)

        print("-" * 80)

print("FINAL ANSWER:")
print(result.get("final_answer"))