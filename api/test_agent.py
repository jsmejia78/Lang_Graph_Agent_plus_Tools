from langgraph_agent import LangGraphAgent
import asyncio
from typing import Dict
import os
import getpass

# Entry point
async def main():
    agent = LangGraphAgent()
    user_message = "Related to machine learning, what is LoRA? Also, who is Tim Dettmers? Also, what is Attention?"

    api_keys  : Dict[str, str] = {
        "OPENAI_API_KEY": getpass.getpass("OpenAI API Key:"),
        "TAVILY_API_KEY": getpass.getpass("TAVILY_API_KEY"),
        "LANGCHAIN_API_KEY": getpass.getpass("LANGCHAIN_API_KEY")
    }
    system_message = "You are a helpful assistant."
    reply = await agent.chat(user_message, api_keys, system_message)

    print(reply["response"])

if __name__ == "__main__":
    asyncio.run(main())