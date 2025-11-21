# LLM Development Examples

A collection of production-quality LLM application examples using LangChain and modern best practices. Covers common patterns like RAG, structured extraction, multi-step agents, and classification.

## 🎯 Overview

This repository contains **reference implementations** of common LLM engineering patterns. All examples use modern **LangChain** approaches (LCEL, structured outputs, composable chains).

## 📁 Examples

### 1. Document Information Extraction
**File:** `document_parser.py`  
**Concepts:** Structured output with Pydantic, validation, error handling

**What it does:**
- Extracts structured data from unstructured customer order text
- Returns validated JSON with customer info, order details, shipping
- Handles missing information gracefully
- Validates data quality (quantities, amounts)

**Key techniques:**
- `with_structured_output()` for reliable extraction
- Field validators for data quality
- Prompt engineering for precise extraction
- Type-safe Pydantic models

---

### 2. RAG Question Answering System
**File:** `rag_qa_system.py`  
**Concepts:** RAG, embeddings, vector stores (Qdrant), LCEL chains

**What it does:**
- Builds Q&A system over store policy documents
- Implements semantic search with Qdrant vector store
- Generates answers with source citations
- Handles out-of-scope questions appropriately

**Key techniques:**
- Document chunking with `RecursiveCharacterTextSplitter`
- OpenAI embeddings
- Qdrant vector store (in-memory mode)
- Modern LCEL RAG chain with `RunnablePassthrough`
- Source document tracking

---

### 3. Multi-Step Analysis Agent
**File:** `multi_step_agent.py`  
**Concepts:** Multi-step reasoning, chained calls, structured workflows

**What it does:**
- Performs order fulfillment assessment workflow
- Three-step workflow: Validation → Fulfillment Assessment → Decision
- Generates detailed reports with reasoning
- Demonstrates complex business logic

**Key techniques:**
- Sequential chain of Pydantic-validated steps
- Complex business rule implementation
- Structured decision-making workflows
- Report generation from structured data

---

### 4. Text Classification & Routing
**File:** `text_classifier.py`  
**Concepts:** Classification, entity extraction, priority queuing

**What it does:**
- Classifies customer support inquiries by type and urgency
- Extracts entities (names, order numbers, amounts)
- Routes to appropriate departments (sales, fulfillment, returns)
- Manages priority queues

**Key techniques:**
- Multi-class classification with Pydantic enums
- Sentiment analysis
- Entity extraction
- Priority-based routing system
- Batch processing

---

### 5. End-to-End Processing Pipeline
**File:** `application_pipeline.py`  
**Concepts:** Integration of multiple patterns, composable systems

**What it does:**
- Complete order processing from raw text to fulfillment decision
- Combines extraction, RAG validation, and multi-step analysis
- Demonstrates pattern integration
- Generates comprehensive reports

**Key techniques:**
- Composing multiple LLM patterns
- RAG-based policy validation
- End-to-end workflow orchestration
- Integrated reporting

---

## 🚀 Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
export OPENAI_API_KEY='your-key-here'
```

### 3. Test Your Setup

```bash
# Run a simple test
python document_parser.py
```

If you see output without errors, you're ready to go!

---

## 📚 How to Use

### Learning:
1. **Study the patterns** - Understand the architectural approaches
2. **Experiment** - Modify examples to test your understanding
3. **Build variations** - Adapt patterns to different use cases
4. **Reference** - Use as templates for new projects

### Key Patterns Demonstrated:
- Modern LCEL chain composition
- Structured outputs with Pydantic
- RAG with Qdrant vector store
- Multi-step agent workflows
- Error handling and validation

---

## 🔑 Key Concepts Demonstrated

### LangChain Components:
- ✅ `ChatOpenAI` - LLM interface
- ✅ `ChatPromptTemplate` - Prompt management
- ✅ `with_structured_output()` - Structured outputs
- ✅ `OpenAIEmbeddings` - Text embeddings
- ✅ `QdrantVectorStore` - Vector store
- ✅ Modern LCEL - RAG chains
- ✅ `RecursiveCharacterTextSplitter` - Document chunking

### Design Patterns:
- ✅ Structured output with Pydantic validation
- ✅ Error handling and graceful degradation
- ✅ Multi-step reasoning workflows
- ✅ Prompt engineering best practices
- ✅ Chain composition
- ✅ Batch processing
- ✅ Confidence scoring

### Production Considerations:
- ✅ Type safety with Pydantic
- ✅ Configurable parameters
- ✅ Clear class structure
- ✅ Comprehensive error handling
- ✅ Logging and debugging support
- ✅ Test data included

---

## 💡 Best Practices Demonstrated

### Code Quality:
- **Modern patterns** - LCEL, structured outputs, composable chains
- **Type safety** - Pydantic models throughout
- **Error handling** - Graceful degradation
- **Documentation** - Clear docstrings and comments

### LangChain Patterns:
- ✅ `with_structured_output()` instead of parsers
- ✅ LCEL chain composition with `|` operator
- ✅ Proper package imports (`langchain_core`, `langchain_community`)
- ✅ Runnable interface for flexibility

---

## 🎓 Concepts to Review

### Before the Interview:

**LangChain Basics:**
- Chain composition (`|` operator)
- Prompt templates and variables
- Output parsers (especially Pydantic)
- Retrieval chains

**LLM Fundamentals:**
- Prompt engineering
- Temperature and sampling
- Token counting and limits
- Structured outputs (JSON mode, function calling)

**RAG Concepts:**
- Chunking strategies
- Embeddings and similarity
- Vector stores (Qdrant)
- Retrieval and ranking
- Modern LCEL composition

**Error Handling:**
- API failures and retries
- Validation errors
- Graceful degradation
- User-friendly error messages

---

## 📝 Quick Reference

### Common LangChain Patterns:

```python
# Simple chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Question: {question}")
chain = prompt | llm
result = chain.invoke({"question": "What is RAG?"})
```

```python
# Structured output (modern approach)
from pydantic import BaseModel

class Answer(BaseModel):
    response: str
    confidence: float

structured_llm = llm.with_structured_output(Answer)
chain = prompt | structured_llm
```

```python
# Modern LCEL RAG chain
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = OpenAIEmbeddings()
vectorstore = QdrantVectorStore.from_texts(
    texts,
    embeddings,
    location=":memory:",
    collection_name="docs"
)

rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 🔧 Troubleshooting

### Common Issues:

**Import errors:**
```bash
pip install --upgrade langchain langchain-openai
```

**API key not found:**
```bash
export OPENAI_API_KEY='your-key'
# Or add to .env file
```

**Qdrant installation:**
```bash
pip install qdrant-client langchain-qdrant
```

**Rate limits:**
- Use `gpt-4o-mini` (faster, cheaper)
- Add retry logic if needed
- Consider local LLM for practice

---

## 📖 Additional Resources

### LangChain Documentation:
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Pydantic Parser](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)
- [RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

### OpenAI Documentation:
- [Function Calling / Structured Outputs](https://platform.openai.com/docs/guides/function-calling)
- [Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

---

## ✨ Using These Examples

**Learning Approach:**
- **Understand the patterns** - Not just the code
- **Experiment** - Modify and test variations
- **Adapt** - Apply patterns to different domains
- **Reference** - Use as templates for new projects

**Key Takeaways:**
- Modern LangChain patterns (LCEL, structured outputs)
- Production-quality error handling
- Composable, maintainable design
- Type-safe implementations

---

## 📫 Notes

- All scripts include sample data and test cases
- Each script is self-contained and runnable
- Code is commented for learning
- Patterns are production-quality but simplified for clarity
- Focus on understanding, not memorization

**Built with:** LangChain, OpenAI, Qdrant  
**Python:** 3.8+

