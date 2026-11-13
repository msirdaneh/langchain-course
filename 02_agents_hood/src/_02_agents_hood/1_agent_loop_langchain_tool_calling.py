from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

from langsmith import traceable

load_dotenv()

MAX_ITERATION = 10
MODEL = "qwen3:8b"
@tool
def get_product_price(product: str) ->float:
    """ Look up the price of a product in the catalog """
    print(f"    >>> Excuting get_product_price(product= '{product}')")
    prices ={
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard" : 89.50
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount tier to a price and return the final price. 
        Available tiers: bronze, silver, gold.s
    """
    print(f"    >>> Execurint apply_discount(price = {price}, discount_tier={discount_tier})")
    discount_percentages = {
        "bronze": 5,
        "sliver": 12,
        "gold" : 23
    }
    
    discount = discount_percentages.get(discount_tier, 0)
    
    return round(price * (1 - discount /100), 2)

# ----- Agent Loop ---- 
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f" Question: {question}")
    print("=" * 60)
    #print (tools_dict)
    messages = [
        SystemMessage(
            content=(
                """ You are a helpful shopping assitant. You have access to a product calling tool
                and a discount tool. \n\n 
                STRICT RULES - you must follow these exactly:
                1. NEVER guess or assume any product price.
                2. You must call get_product_price first to get the real prices.
                3. Only call apply_discount AFTER you have received a price from get_product_price
                 Pass the exact price returned by get_product_price - do NOT pass a made-up number.
                4. NEVER calculate discounts yourself using math.
                5. If user does not specify a discount tier, as them which tier to use - do NOT assume one
                """
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n --- Iteration {iteration} ----")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        # if not toola, this is the final answer
        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content
        # process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tools Selected] {tool_name} with args {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name) 

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: Max iteration reached witout a final answer")
    return None




if __name__ == "__main__":
    print("Hello Langchain with binging tools")
    print()
    result = run_agent("What is the proce of the laptop after applyig a gold discount")