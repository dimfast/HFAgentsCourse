import asyncio
import llama_index.core
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

llama_index.core.set_global_handler("simple")


def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    print("----> Multiplication tool called!")
    return a * b


llm = Ollama(model="qwen2.5:3b", request_timeout=60.0, temperature=0)

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)


async def main():
    response = await agent.run("What is 1235 times 32?")
    print(response)


asyncio.run(main())

