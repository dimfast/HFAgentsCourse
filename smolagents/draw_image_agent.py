import os
from PIL import Image

from smolagents import CodeAgent, InferenceClientModel, FinalAnswerTool, load_tool


hf_token = os.getenv("HF_TOKEN")
model = InferenceClientModel(
    model_id="moonshotai/Kimi-K2.5",
    token=hf_token
)
final_answer = FinalAnswerTool()

image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

agent = CodeAgent(
    tools=[image_generation_tool, final_answer],
    model=model,
    add_base_tools=False,
    max_steps=3,
    verbosity_level=2,
)

response = agent.run("Draw cute orange cat in futuristic exosuit.")

print("\n\n")
if isinstance(response, Image.Image):
    response.save("result.png")
    print("Agent returned an image. Saved it to 'result.png'.")
else:
    print("Agent returned text answer:", response)
