from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
import os

import langchain_course
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilyResearch

load_dotenv()

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the internet

#     Args:
#         query: The query to search for
#     Returns:
#         the search result 
#     """
#     print(f"seraching for {query}")
#     return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-4.1-mini")
# tools =[search]
tools = [TavilyResearch()]

agent = create_agent(model=llm, tools=tools)

def main():
    query = """
            search for 3 job posting for an AI engineer in the Saudi Arabia on linkedin and list their details
    """
    message = HumanMessage(content=query)

    result = agent.invoke({"messages": message})
    
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()