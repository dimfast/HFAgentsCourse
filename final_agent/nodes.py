from pathlib import Path
import requests
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    task_id: str
    question: str
    file_name: str
    file_path: Optional[str]
    task_type: str
    answer_format: str
    final_answer: str
    active_task_node: str
    messages: Annotated[list[BaseMessage], add_messages]


BASE_URL = "https://agents-course-unit4-scoring.hf.space"

DOWNLOADS_DIR = Path(__file__).resolve().parent / "downloads"


TASK_CATEGORIES = {
    "web",
    "youtube",
    "audio",
    "image",
    "excel",
    "python_file",
    "calculation",
    "logic",
    "simple",
    "unknown",
}


CLASSIFY_PROMPT = """
Classify the task into exactly one category.

Categories:
web - needs web search or webpage reading
youtube - needs YouTube video/audio
audio - needs attached audio file
image - needs attached image
excel - needs attached Excel file
python_file - needs attached Python file
calculation - needs arithmetic/table computation
logic - needs reasoning, puzzle, text manipulation
simple - can be answered directly
unknown - unclear

Return only one category name. No explanation or any other text.
"""


WEB_PROMPT = """
You answer GAIA-style questions using tools.

Use web_search first when needed.
Use fetch_webpage for promising URLs.

Never output thoughts, reasoning, JSON, plans, or explanations.
Never return {"thought": "..."}.
When you need more information, call a tool.
When you know the answer, return only the final answer.

No "FINAL ANSWER:" prefix.
No markdown.
No extra text.
"""


PYTHON_PROMPT = """
You answer questions involving attached Python files or Python execution.

Use read_text_file to inspect attached files.
Use python_exec when calculation or code execution is needed.
Return only the final answer.
No explanation.
No "FINAL ANSWER:" prefix.
"""


REASONING_PROMPT = """
You answer reasoning, logic, and calculation questions.

Use python_exec when it helps avoid mistakes.
Return only the final answer.
No explanation.
No "FINAL ANSWER:" prefix.
"""


def classify_by_rules(question: str, file_name: str) -> str:
    q = question.lower()
    f = (file_name or "").lower()

    if "youtube.com" in q or "youtu.be" in q:
        return "youtube"
    if f.endswith(".mp3"):
        return "audio"
    if f.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    if f.endswith(".xlsx"):
        return "excel"
    if f.endswith(".py"):
        return "python_file"
    if "wikipedia" in q or "article" in q or "paper" in q:
        return "web"
    if "calculate" in q or "how many" in q or "table" in q:
        return "calculation"
    if "opposite" in q or "reverse" in q or "logically" in q:
        return "logic"

    return "simple"


def classify_task(llm, state):
    file_name = state.get("file_name") or ""

    prompt = f"Question:\n{state['question']}\n\nAttached file:\n{file_name or 'None'}"

    try:
        response = llm.invoke([
            SystemMessage(content=CLASSIFY_PROMPT),
            HumanMessage(content=prompt),
        ])

        category = response.content.strip().lower()

        if category not in TASK_CATEGORIES:
            category = "unknown"

    except Exception:
        category = classify_by_rules(state["question"], file_name)

    return {"task_type": category}


def download_task_file(task_id: str, file_name: str) -> str:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/files/{task_id}"
    path = DOWNLOADS_DIR / Path(file_name).name

    if path.exists():
        return str(path)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    path.write_bytes(response.content)
    return str(path)


def prepare_file(state):
    file_name = state.get("file_name") or ""

    if not file_name:
        return {"file_path": None}

    try:
        file_path = download_task_file(state["task_id"], file_name)
        return {"file_path": file_path}
    except Exception as exc:
        return {
            "file_path": None,
            "messages": [
                AIMessage(content=f"ERROR: failed to download attached file: {exc}")
            ],
        }


def web_task(llm_with_tools, state):
    if state["messages"]:
        messages = state["messages"]
    else:
        messages = [
            SystemMessage(content=WEB_PROMPT),
            HumanMessage(content=state["question"]),
        ]

    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_task_node": "web_task",
    }


def python_task(llm_with_tools, state):
    messages = state["messages"]

    if not messages:
        file_context = f"Attached file path: {state.get('file_path') or 'None'}"
        messages = [
            SystemMessage(content=PYTHON_PROMPT),
            HumanMessage(content=f"{state['question']}\n\n{file_context}"),
        ]

    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_task_node": "reasoning_task",
    }


def reasoning_task(llm_with_tools, state):
    messages = state["messages"]

    if not messages:
        messages = [
            SystemMessage(content=REASONING_PROMPT),
            HumanMessage(content=state["question"]),
        ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def make_nodes(llm, tools):
    llm_with_tools = llm.bind_tools(tools)

    return {
        "classify_task": lambda state: classify_task(llm, state),
        "prepare_file": prepare_file,
        "web_task": lambda state: web_task(llm_with_tools, state),
        "python_task": lambda state: python_task(llm_with_tools, state),
        "reasoning_task": lambda state: reasoning_task(llm_with_tools, state),
    }


def normalize_answer(text: str) -> str:
    text = text.strip()

    prefixes = [
        "FINAL ANSWER:",
        "Final answer:",
        "final answer:",
        "The final answer is",
        "the final answer is",
        "Answer:",
        "ANSWER:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.strip()
    text = text.strip('"').strip("'").strip()

    if text.endswith("."):
        text = text[:-1].strip()

    return text


def finalize_answer(state):
    raw = state["messages"][-1].content
    answer = normalize_answer(raw)
    return {"final_answer": answer}
