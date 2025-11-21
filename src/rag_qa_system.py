"""
RAG-based Question Answering System
====================================
A retrieval-augmented generation system for Q&A over documents.

Features:
- Document loading and chunking
- Semantic search with Qdrant vector store
- Modern LCEL chain composition
- Source citation support
- Graceful handling of out-of-scope questions
"""

import os
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Sample policy documents (in real scenario, would load from files)
# Using store policies as example, but works for any policy domain
SAMPLE_POLICIES = """
ONLINE STORE POLICY MANUAL

Section 1: Shipping Policy
==========================
1.1 Shipping Methods:
- Standard Shipping: 5-7 business days ($5.99 flat rate)
- Express Shipping: 2-3 business days ($12.99 flat rate)
- Overnight Shipping: Next business day ($24.99 flat rate)
- Free shipping on orders over $50

1.2 International Shipping:
- Available to Canada and Mexico
- Delivery time: 10-15 business days
- Customs fees are the responsibility of the customer
- Minimum order value: $25

1.3 Order Processing:
- Orders placed before 2 PM EST ship same day
- Orders placed after 2 PM EST ship next business day
- No shipping on weekends or holidays
- Tracking number provided within 24 hours of shipment

Section 2: Return Policy
========================
2.1 Return Window:
- 30 days from delivery date for most items
- 14 days for opened electronics
- 60 days for defective items (with proof)
- Holiday purchases: Extended until January 31st

2.2 Return Conditions:
- Items must be in original packaging
- All accessories and documentation included
- No signs of use or wear
- Original receipt or order confirmation required

2.3 Refund Process:
- Refunds processed within 5-7 business days
- Original payment method only
- Shipping costs non-refundable (except for defects)
- Restocking fee of 15% may apply to opened items

2.4 Non-Returnable Items:
- Personalized or custom items
- Digital downloads
- Intimate apparel
- Sale items marked "Final Sale"

Section 3: Product Information
==============================
3.1 Electronics:
- 1-year manufacturer warranty
- Free technical support via phone/email
- Software updates available on website
- Compatible with standard US power outlets

3.2 Furniture:
- Assembly required unless noted
- Assembly instructions included
- Dimensions listed in product description
- Weight capacity specified for chairs/desks

3.3 Apparel:
- Size chart available on each product page
- Materials and care instructions on label
- Color may vary slightly from photos
- Pre-shrunk cotton items

Section 4: Order Modification
==============================
4.1 Before Shipping:
- Order can be canceled within 1 hour of placement
- Address changes accepted before processing
- Item substitutions possible if in stock
- Contact customer service immediately

4.2 After Shipping:
- Cannot cancel once shipped
- Redirect to new address may incur fees
- Return and reorder if changes needed

Section 5: Customer Service
============================
5.1 Contact Methods:
- Phone: 1-800-SHOP-NOW (M-F 9 AM-8 PM EST)
- Email: support@onlinestore.com (24-48 hour response)
- Live Chat: Available on website (M-F 9 AM-6 PM EST)
- Mail: Customer Service, 123 Commerce Dr, City, ST 12345

5.2 Response Times:
- Urgent issues: Same business day
- General inquiries: Within 24-48 hours
- Complex issues: 3-5 business days

5.3 Issue Resolution Priority:
- Order not received: URGENT
- Defective item: HIGH
- General questions: NORMAL
- Product suggestions: LOW


"""


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
    
    def load_documents(self, documents: List[str] or str) -> None:
        """
        Load and process documents into vector store
        
        Args:
            documents: Single document string or list of documents
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
    rag.load_documents(SAMPLE_POLICIES)
    
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
    rag_advanced.load_documents(SAMPLE_POLICIES)
    
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

