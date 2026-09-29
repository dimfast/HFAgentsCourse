import datetime
import os

import pytz
from smolagents import tool, CodeAgent, InferenceClientModel, FinalAnswerTool


@tool
def get_current_time(timezone: str) -> str:
    """
    A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz=tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


hf_token = os.getenv("HF_TOKEN")
model = InferenceClientModel(
    model_id="moonshotai/Kimi-K2.5",
    token=hf_token
)
final_answer = FinalAnswerTool()

agent = CodeAgent(
    tools=[get_current_time, final_answer],
    model=model,
    add_base_tools=False,
    max_steps=3,
    verbosity_level=2,
)

response = agent.run("What is the current time in Amsterdam?")

print("\n\nFinal response:")
print(response)