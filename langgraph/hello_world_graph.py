import io
from typing import TypedDict
from PIL import Image
import matplotlib.pyplot as plt

from langgraph.graph import StateGraph


class AgentState(TypedDict):
    text: str


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state."""
    state["text"] = "Hey " + state["text"] + ", how are you?"
    return state


graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)

graph.set_entry_point("greeting")
graph.set_finish_point("greeting")

app = graph.compile()

img = Image.open(io.BytesIO(app.get_graph().draw_mermaid_png()))
# img.save("hello_world_agent.png")
# plt.figure(figsize=(10, 8))
# plt.imshow(img)
# plt.axis('off')
# plt.show()

result = app.invoke({"text": "Dima"})

print(type(result))
print(result["text"])