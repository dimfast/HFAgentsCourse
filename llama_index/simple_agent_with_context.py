import asyncio
import llama_index.core
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context

llama_index.core.set_global_handler("simple")


llm = Ollama(model="qwen2.5:3b", request_timeout=60.0, temperature=0)

agent = AgentWorkflow.from_tools_or_functions([], llm=llm)


async def main():
    ctx = Context(agent)
    response = await agent.run("My name is Bob.", ctx=ctx)
    print("\tResponse 1:", response)
    response = await agent.run("What was my name again?", ctx=ctx)
    print("\tResponse 2:", response)


asyncio.run(main())

