"""
RAG-based Question Answering System
====================================
A retrieval-augmented generation system for Q&A over documents.

Features:
- Document loading from text files using LangChain TextLoader
- Semantic search with Qdrant vector store
- Modern LCEL chain composition
- Source citation support
- Graceful handling of out-of-scope questions
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()


class PolicyRAG:
    """RAG system for answering questions about company policies"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG system
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for generation
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self, documents: Union[str, List[str]]) -> None:
        """
        Load and process documents into vector store
        
        Args:
            documents: Single document string or list of documents (raw text)
        """
        if isinstance(documents, str):
            documents = [documents]
        
        # Create Document objects
        docs = [Document(page_content=doc) for doc in documents]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        print(f"📄 Loaded {len(documents)} document(s)")
        print(f"✂️  Split into {len(chunks)} chunks")
        
        # Create Qdrant vector store (in-memory for simplicity)
        self.vectorstore = QdrantVectorStore.from_documents(
            chunks,
            self.embeddings,
            location=":memory:",  # In-memory for interview/testing
            collection_name="policies"
        )
        print(f"✅ Vector store created")
        
        # Create modern LCEL RAG chain
        self._create_rag_chain()
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load document from a text file using LangChain TextLoader
        
        Args:
            file_path: Path to text file to load
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        # Use TextLoader to load the document
        loader = TextLoader(str(path))
        docs = loader.load()
        
        print(f"📄 Loaded: {path.name}")
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        print(f"✂️  Split into {len(chunks)} chunks")
        
        # Create Qdrant vector store (in-memory for simplicity)
        self.vectorstore = QdrantVectorStore.from_documents(
            chunks,
            self.embeddings,
            location=":memory:",
            collection_name="policies"
        )
        print(f"✅ Vector store created")
        
        # Create modern LCEL RAG chain
        self._create_rag_chain()
    
    def _create_rag_chain(self) -> None:
        """Create modern LCEL RAG chain"""
        
        # Create prompt template
        prompt_template = """You are a policy expert. Use the following policy information to answer the question accurately.

If the answer is not in the provided context, say "I don't have that information in the current policy documents." Do not make up information.

Always cite which section of the policy your answer comes from.

Context:
{context}

Question: {question}

Answer with citations:"""
        
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        # Build modern LCEL chain
        def format_docs(docs):
            """Format retrieved documents"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str) -> Dict[str, str | List[Document]]:
        """
        Ask a question and get an answer with sources
        
        Args:
            question: Question about store policies
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.rag_chain is None:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        # Get answer using modern RAG chain
        answer = self.rag_chain.invoke(question)
        
        # Get source documents separately for citations
        source_docs = self.retriever.invoke(question)
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_docs
        }
    
    def ask_with_sources(self, question: str) -> str:
        """
        Ask question and format answer with source citations
        
        Args:
            question: Question about store policies
            
        Returns:
            Formatted answer with sources
        """
        result = self.ask(question)
        
        # Format output
        output = f"Question: {result['question']}\n\n"
        output += f"Answer: {result['answer']}\n\n"
        output += "Sources:\n"
        
        for i, doc in enumerate(result['source_documents'], 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            output += f"  [{i}] {preview}...\n"
        
        return output
    
    def batch_questions(self, questions: List[str]) -> List[Dict]:
        """
        Answer multiple questions efficiently
        
        Args:
            questions: List of questions
            
        Returns:
            List of results
        """
        results = []
        for question in questions:
            try:
                result = self.ask(question)
                results.append(result)
            except Exception as e:
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "source_documents": []
                })
        return results


class PolicyRAGAdvanced(PolicyRAG):
    """Advanced RAG with query routing and filtering"""
    
    def ask_with_confidence(self, question: str) -> Dict:
        """
        Answer with confidence assessment
        
        Returns:
            Dict with answer, confidence level, and reasoning
        """
        # Get standard answer
        result = self.ask(question)
        
        # Assess confidence
        confidence_prompt = f"""Review this Q&A interaction. Assess the confidence level of the answer.

Question: {result['question']}
Answer: {result['answer']}

Consider:
1. How directly the sources answer the question
2. Whether information might be incomplete
3. If assumptions were made

Provide:
- Confidence: HIGH/MEDIUM/LOW
- Reasoning: Brief explanation

Format as: CONFIDENCE: [level] | REASONING: [explanation]"""
        
        confidence_response = self.llm.invoke(confidence_prompt)
        
        return {
            **result,
            "confidence_assessment": confidence_response.content
        }
    
    def search_similar(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar content without generating an answer
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("No documents loaded.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results


def main():
    """Demonstrate the RAG system"""
    
    print("="*60)
    print("Policy RAG System Demo")
    print("="*60)
    
    # Initialize RAG system
    print("\n1. Initializing RAG system...")
    rag = PolicyRAG()
    
    # Load policy documents
    print("\n2. Loading policy documents...")
    
    # Determine file path
    import sys
    if len(sys.argv) > 1:
        # Load from file path provided as command line argument
        file_path = sys.argv[1]
    else:
        # Load from default sample file
        # Get project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        file_path = project_root / "docs" / "sample_policies.txt"
    
    print(f"Loading from: {file_path}")
    rag.load_from_file(str(file_path))
    
    # Test questions
    test_questions = [
        "What are the shipping options available?",
        "What is the return policy for electronics?",
        "How long do I have to return an item?",
        "Do you offer free shipping?",
        "What items cannot be returned?",
        "How do I contact customer service?",
        "What is the weather like today?",  # Out of scope - should handle gracefully
    ]
    
    print("\n3. Testing Q&A System:")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}]")
        print(rag.ask_with_sources(question))
        print("-"*60)
    
    # Test advanced features
    print("\n4. Testing Advanced Features:")
    print("="*60)
    
    rag_advanced = PolicyRAGAdvanced()
    rag_advanced.load_from_file(str(file_path))
    
    test_question = "What are the conditions for returning an item?"
    print(f"\nQuestion: {test_question}")
    result = rag_advanced.ask_with_confidence(test_question)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nConfidence Assessment:\n{result['confidence_assessment']}")
    
    # Test similarity search
    print("\n5. Testing Semantic Search:")
    print("="*60)
    search_query = "shipping information"
    print(f"\nSearching for: '{search_query}'")
    similar_docs = rag_advanced.search_similar(search_query, k=3)
    
    for i, (doc, score) in enumerate(similar_docs, 1):
        print(f"\n[Result {i}] Similarity: {score:.3f}")
        print(f"{doc.page_content[:200]}...")
    
    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        main()

