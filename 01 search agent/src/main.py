import os
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilyResearch

load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """ Schema for a source used by the agent"""
    url: str =Field(description="The url for the source")

class AgentResponse(BaseModel):
    """ schema for the agent response """
    answer: str = Field(description= "The agent answer to the query")
    sources: List[Source] = Field(default_factory= list, description= " The ist of sources used to generate the answer")


@tool
def search(query: str) -> str:
    """
    Tool that searches the internet

    Args:
        query: The query to search for
    Returns:
        the search result 
    """
    print(f"seraching for {query}")
    return tavily.search(query=query)

# print(f"Tavily API key: {os.getenv("TAVILY_API_KEY")}")
llm = ChatOpenAI(model="gpt-4.1-mini")
tools =[search]
# tools = [TavilyResearch()]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    

    query = """
            search for 3 job posting for an AI engineer in the Saudi Arabia on linkedin and list their details
    """
    message = HumanMessage(content=query)
    
    try:
        result = agent.invoke(
            {"messages": message},
            )
        # print(result["messages"][-1].content)
        # in case of structured output
        print(result["structured_response"].answer)
        print(result["structured_response"].sources)
    except Exception as e:
        print(type(e))
        print(e)

    

if __name__ == "__main__":
    main()