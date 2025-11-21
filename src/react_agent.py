"""
Simple LangGraph 1.0 Agent using local Ollama LLM
Optimized for 16GB GPU
With LangSmith tracing
"""

import os
import warnings
import time
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from dotenv import load_dotenv

load_dotenv()

# Suppress LangSmith UUID v7 warning (this is a deprecation warning from pydantic v1)
warnings.filterwarnings("ignore", message=".*LangSmith now uses UUID v7.*")

#Set the model type to "ollama" or "openai"
# model_type = "ollama"
model_type = "openai"
# Set your preferred model here
# model_name = "llama3.2:3b"
model_name = "gpt-4o-mini"

# Set project name for LangSmith tracing
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_PROJECT"] = f"{model_name}-agent-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"📊 LangSmith Project: {os.environ.get('LANGCHAIN_PROJECT')}")

# Define some simple tools for the agent
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Example: '2 + 2' or '10 * 5'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_word_length(word: str) -> str:
    """Returns the length of a given word."""
    return f"The word '{word}' has {len(word)} letters."


@tool
def reverse_string(text: str) -> str:
    """Reverses the given text."""
    return f"Reversed: {text[::-1]}"


# Initialize LLM with the selected model
if model_type == "ollama":
    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )
else:
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

# Bind tools to the LLM
tools = [calculator, get_word_length, reverse_string]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Define the agent node
def call_agent(state: MessagesState) -> MessagesState:
    """Call the agent with the current state."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Define the tool execution node
def tool_node(state: MessagesState) -> MessagesState:
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Define the routing function
def should_continue(state: MessagesState) -> str:
    """Determine whether to continue or end the agent loop."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are no tool calls, we're done
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# Build the graph
def create_agent():
    """Create and compile the LangGraph agent."""
    # Create the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    return app

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. Your role is to:

1. Analyze the user's request carefully
2. Think step-by-step through problems before acting.
3. Use available tools when needed to provide accurate answers. 
4. If the user's request is not related to the tools, answer the question directly.
5. Provide clear, concise responses

Available tools:
- calculator: For mathematical computations
- get_word_length: To count letters in words
- reverse_string: To reverse text

**CRITICAL**:
ONLY call a tool if:
    - A calculation is required (use calculator)
    - The user explicitly asks to count letters (use get_word_length)  
    - The user explicitly asks to reverse text (use reverse_string)

**IMPORTANT**:
- Do not make up information. If you don't know the answer, say so.
- Do not hallucinate. If you don't know the answer, say so.
- Do not make up information. If you don't know the answer, say so.
"""

def run_agent(user_input: str):
    """Run the agent with a user input."""
    app = create_agent()
    
    # Create initial state with system prompt and user message
    initial_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
    }
    
    # Run the agent
    print(f"\n🤖 User: {user_input}\n")
    print("=" * 60)
    
    for event in app.stream(initial_state):
        for key, value in event.items():
            if key == "agent":
                last_message = value["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    print(f"\n💭 Agent: {last_message.content}")
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    print(f"\n🔧 Tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
            elif key == "tools":
                print(f"\n📊 Tool results: ", end="")
                for msg in value["messages"]:
                    if isinstance(msg, ToolMessage):
                        print(f"{msg.content}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Test examples
    examples = [
        "What is 25 multiplied by 4?",
        "How many letters are in the word 'artificial'?",
        "Can you reverse the text 'Hello World'?",
        "What is the capital of France?",
    ]
    
    print("Running example queries...\n")
    
    for example in examples:
        run_agent(example)
        print("\n")
