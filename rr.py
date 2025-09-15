from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
#from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.sync import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------
# 2. Tools - Fixed to use consistent naming
# -------------------

# Create custom search tool with explicit name
@tool("brave_search")
def brave_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for current information.
    Use this when you need to find recent information or answer questions about current events.
    """
    search_tool = DuckDuckGoSearchRun(region="us-en")
    try:
        result = search_tool.run(query)
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool("calculator")
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {
            "first_num": first_num, 
            "second_num": second_num, 
            "operation": operation, 
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

@tool("get_stock_price")
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage API.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=02N8BQIZNVL0B3RU"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# List of tools
tools = [brave_search, calculator, get_stock_price]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        # Handle any API errors gracefully
        error_msg = f"Error in chat_node: {str(e)}"
        print(error_msg)  # Log for debugging
        # Return a simple text response instead of failing
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=f"I apologize, but I encountered an error: {error_msg}")]}

# Create tool node
tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
try:
    conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
except Exception as e:
    print(f"Warning: Could not initialize checkpointer: {e}")
    checkpointer = None

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile graph with or without checkpointer
if checkpointer:
    chatbot = graph.compile(checkpointer=checkpointer)
else:
    chatbot = graph.compile()

# -------------------
# 7. Helper Functions
# -------------------
def retrieve_all_threads():
    """Retrieve all conversation threads from the database."""
    if not checkpointer:
        return []
    
    try:
        all_threads = set()
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
        return list(all_threads)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        return []

