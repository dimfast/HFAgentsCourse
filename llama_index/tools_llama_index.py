from llama_index.core.tools import FunctionTool

def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
result = tool.call("New York")
print(result)


from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

print([(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list])