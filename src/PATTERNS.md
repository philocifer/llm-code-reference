# LangChain Pattern Reference

Quick reference for common LangChain patterns and idioms.

---

## 🔧 Basic Setup (Start Every Script)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Fast and cheap
    temperature=0.0,       # Deterministic for production
    api_key=os.getenv("OPENAI_API_KEY")
)
```

**Alternative: Local LLM with Ollama**

```python
from langchain_ollama import ChatOllama

# For local models (requires Ollama installed)
llm = ChatOllama(
    model="llama3.2:3b",  # or other Ollama models
    temperature=0.0,
)
```

---

## 📝 Pattern 1: Simple Prompt Chain

```python
from langchain_core.prompts import ChatPromptTemplate

# Create prompt
prompt = ChatPromptTemplate.from_template(
    "You are a {role}. {instruction}\n\nInput: {input}"
)

# Create chain
chain = prompt | llm

# Run
result = chain.invoke({
    "role": "customer service agent",
    "instruction": "Answer this inquiry",
    "input": inquiry_text
})

print(result.content)
```

---

## 📊 Pattern 2: Structured Output (with_structured_output)

```python
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Define schema
class OrderData(BaseModel):
    customer_name: str = Field(description="Full name")
    email: Optional[str] = Field(description="Email address", default=None)
    product_name: str = Field(description="Product ordered")
    quantity: int = Field(description="Quantity", ge=1, le=100)

# Create structured LLM
structured_llm = llm.with_structured_output(OrderData)

# Create prompt (no format instructions needed!)
prompt = ChatPromptTemplate.from_template(
    "Extract order information from the text.\n\nText: {text}"
)

# Chain
chain = prompt | structured_llm

# Run
result = chain.invoke({"text": raw_text})

# result is now an OrderData object
print(result.customer_name)
print(result.product_name)
```

**Why this is better:**
- ✅ Uses OpenAI's native structured output (more reliable)
- ✅ No format instructions needed (cleaner prompts)
- ✅ Better error handling
- ✅ Faster and more accurate

**Advanced: Custom Validators**

```python
from pydantic import BaseModel, Field, field_validator

class OrderData(BaseModel):
    customer_name: str = Field(description="Full name")
    quantity: int = Field(description="Quantity", ge=1)
    unit_price: float = Field(description="Price per unit")
    
    @field_validator('unit_price')
    @classmethod
    def validate_positive_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v > 1000:
            raise ValueError('Quantity cannot exceed 1000')
        return v
```

---

## 🔍 Pattern 3: RAG (Q&A over Documents) - MODERN LCEL

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = [Document(page_content=text) for text in documents]
chunks = splitter.split_documents(docs)

# 2. Create embeddings and Qdrant vector store
embeddings = OpenAIEmbeddings()
vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embeddings,
    location=":memory:",  # Or use URL for persistent storage
    collection_name="my_documents"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Create prompt
prompt = ChatPromptTemplate.from_template(
    """Answer based on context below.
    
Context: {context}

Question: {question}

Answer:"""
)

# 4. Build modern LCEL RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask questions
answer = rag_chain.invoke("What are the requirements?")
print(answer)

# Get source documents separately if needed
source_docs = retriever.invoke("What are the requirements?")
```

**Why this is better:**
- ✅ Modern LCEL composition (future-proof)
- ✅ More flexible and composable
- ✅ Explicit control over each step
- ✅ No deprecated chains

---

## 🔗 Pattern 4: Multi-Step Workflow

```python
# Define steps with structured outputs
class Step1Output(BaseModel):
    field1: str

class Step2Output(BaseModel):
    field2: int

# Step 1
def step1(input_data):
    structured_llm = llm.with_structured_output(Step1Output)
    prompt = ChatPromptTemplate.from_template(
        "Process this data.\n\nData: {data}"
    )
    chain = prompt | structured_llm
    return chain.invoke({"data": input_data})

# Step 2 (uses Step 1 output)
def step2(step1_result):
    structured_llm = llm.with_structured_output(Step2Output)
    prompt = ChatPromptTemplate.from_template(
        "Based on: {previous}, provide analysis."
    )
    chain = prompt | structured_llm
    return chain.invoke({"previous": step1_result.field1})

# Run workflow
result1 = step1(raw_input)
result2 = step2(result1)
```

---

## 🎯 Pattern 5: Classification

```python
from typing import Literal
from enum import Enum

class Category(str, Enum):
    TYPE_A = "type_a"
    TYPE_B = "type_b"
    TYPE_C = "type_c"

class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

structured_llm = llm.with_structured_output(Classification)
prompt = ChatPromptTemplate.from_template(
    """Classify the following text.
    
Categories:
- type_a: Description
- type_b: Description  
- type_c: Description

Text: {text}"""
)

chain = prompt | structured_llm
result = chain.invoke({"text": input_text})
```

**Enum vs Literal:**
- Use `Enum` for reusable categories across your codebase
- Use `Literal` for simple, one-off constrained strings:

```python
from typing import Literal

class Analysis(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    priority: Literal["low", "medium", "high"]
```

---

## 🤖 Pattern 6: LangGraph Agent with Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

# Create tool execution mapping
tools_by_name = {tool.name: tool for tool in tools}

# Define agent node
def call_agent(state: MessagesState) -> MessagesState:
    """Call the agent with the current state"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Define tool execution node
def tool_node(state: MessagesState) -> MessagesState:
    """Execute tool calls"""
    from langchain_core.messages import ToolMessage
    
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Define routing
def should_continue(state: MessagesState) -> str:
    """Determine whether to continue or end"""
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

# Compile and run
app = workflow.compile()

# Use the agent
system_prompt = "You are a helpful assistant with access to tools."
user_input = "What is 25 * 4?"

# Option 1: Get final result
result = app.invoke({
    "messages": [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
})

# Option 2: Stream intermediate results
for event in app.stream({"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]}):
    for key, value in event.items():
        if key == "agent":
            print(f"Agent: {value['messages'][-1].content}")
        elif key == "tools":
            print(f"Tool results: {value['messages'][-1].content}")
```

**Why use LangGraph:**
- ✅ Full control over agent flow
- ✅ Tool calling with iterative refinement
- ✅ State management for complex workflows
- ✅ Can stream intermediate results
- ✅ More flexible than simple chains

---

## ⚠️ Error Handling

```python
def safe_llm_call(chain, input_data, max_retries=3):
    """Wrapper for LLM calls with retry logic"""
    for attempt in range(max_retries):
        try:
            result = chain.invoke(input_data)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
# Use it
try:
    result = safe_llm_call(chain, {"input": data})
except Exception as e:
    print(f"Failed after retries: {e}")
    # Handle gracefully
```

---

## 🔎 Vector Search (Without Full RAG)

```python
# Just semantic search, no LLM
embeddings = OpenAIEmbeddings()
vectorstore = QdrantVectorStore.from_texts(
    documents,
    embeddings,
    location=":memory:",
    collection_name="search"
)

# Search
results = vectorstore.similarity_search(query, k=5)
for doc in results:
    print(doc.page_content)

# With scores
results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
for doc, score in results_with_scores:
    print(f"Score: {score:.3f} - {doc.page_content[:100]}")
```

---

## 📦 Custom Prompt Templates

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

system_template = "You are a {role} with expertise in {domain}."
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{instruction}\n\nInput: {input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    human_prompt
])

chain = chat_prompt | llm

result = chain.invoke({
    "role": "support specialist",
    "domain": "e-commerce",
    "instruction": "Answer this customer question",
    "input": inquiry_text
})
```

---

## 🧪 Quick Testing Pattern

```python
def main():
    # Test data
    test_cases = [
        "Case 1 input",
        "Case 2 input",
        "Edge case input"
    ]
    
    # Initialize
    processor = MyProcessor()
    
    # Test each
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        
        try:
            result = processor.process(test_input)
            print(f"✅ Success")
            print(f"Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

---

## 💾 Persistent Vector Store

```python
# In-memory (for testing/interview)
vectorstore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:",
    collection_name="my_docs"
)

# Persistent local storage
vectorstore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    path="./qdrant_data",  # Local disk storage
    collection_name="my_docs"
)

# Production: Connect to Qdrant server
vectorstore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url="http://localhost:6333",  # Or cloud URL
    collection_name="my_docs"
)
```

---

## 🎨 Pretty Output Formatting

```python
def format_result(data: dict) -> str:
    """Format structured data nicely"""
    output = []
    output.append("="*60)
    output.append("RESULT")
    output.append("="*60)
    
    for key, value in data.items():
        if isinstance(value, list):
            output.append(f"\n{key.replace('_', ' ').title()}:")
            for item in value:
                output.append(f"  • {item}")
        else:
            output.append(f"{key.replace('_', ' ').title()}: {value}")
    
    output.append("="*60)
    return "\n".join(output)

print(format_result(result.model_dump()))
```

---

## 🚀 Starting Template for Any Problem

```python
"""
Problem: [Description]
Time: ~XX minutes
"""

import os
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class OutputSchema(BaseModel):
    """Define your output structure"""
    field1: str = Field(description="Description")
    field2: int = Field(description="Description")


class Processor:
    """Main processor class"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.structured_llm = self.llm.with_structured_output(OutputSchema)
    
    def process(self, input_data: str) -> OutputSchema:
        """Process input and return structured output"""
        prompt = ChatPromptTemplate.from_template(
            """Your instruction here.

Input: {input}

Process the input:"""
        )
        
        chain = prompt | self.structured_llm
        
        return chain.invoke({"input": input_data})


def main():
    """Test the processor"""
    processor = Processor()
    
    test_input = "Sample input"
    result = processor.process(test_input)
    
    print(result)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        main()
```

**Note**: Always check for the API key before running to avoid cryptic errors.

---

## 🐛 Debug Tips

```python
# Option 1: LangSmith tracing (recommended for production debugging)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"
# Now all LLM calls will be traced in LangSmith

# Option 2: Local debug mode
import langchain
langchain.debug = True  # Shows all LLM calls and prompts in console

# Option 3: Print intermediate results
result = step1(input)
print(f"Step 1 output: {result}")  # Debug print
result = step2(result)

# Option 4: Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*specific warning text.*")
```

---

## ⚡ Quick Development Tips

**Efficient workflow:**
1. Start with clear requirements
2. Set up data models first (Pydantic)
3. Build core functionality
4. Add error handling
5. Test with real examples

---

## 🎯 Best Practices

- ✅ Simple, clear code is better than complex code
- ✅ Test incrementally as you build
- ✅ Use logging/prints for debugging
- ✅ Handle errors gracefully
- ✅ Comment complex logic
- ✅ Use type hints
- ✅ Reference documentation when needed

