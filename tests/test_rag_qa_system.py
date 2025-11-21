"""
Pytest Tests for RAG QA System (Real API Calls)
================================================
Tests PolicyRAG and PolicyRAGAdvanced with actual API.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_qa_system import PolicyRAG, PolicyRAGAdvanced


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_policy():
    """Sample policy document"""
    return """
    COMPANY POLICY MANUAL
    
    Shipping Policy:
    - Standard shipping: 5-7 business days, $5.99
    - Express shipping: 2-3 business days, $12.99
    - Free shipping on orders over $50
    
    Return Policy:
    - Returns accepted within 30 days
    - Items must be unopened and in original packaging
    - Refunds processed in 5-7 business days
    - Restocking fee of 15% for opened electronics
    
    Customer Service:
    - Phone support: M-F 9 AM-6 PM EST
    - Email: support@company.com (24-48 hour response)
    - Live chat available on website
    """


@pytest.fixture
def rag_system(sample_policy):
    """Create RAG system with loaded documents"""
    rag = PolicyRAG()
    rag.load_documents(sample_policy)
    return rag


@pytest.fixture
def rag_advanced(sample_policy):
    """Create advanced RAG system"""
    rag = PolicyRAGAdvanced()
    rag.load_documents(sample_policy)
    return rag


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Test RAG system initialization"""
    
    def test_basic_rag_creation(self):
        """Test PolicyRAG creates successfully"""
        rag = PolicyRAG()
        assert rag is not None
        assert rag.llm is not None
        assert rag.embeddings is not None
        assert rag.text_splitter is not None
    
    def test_advanced_rag_creation(self):
        """Test PolicyRAGAdvanced creates successfully"""
        rag = PolicyRAGAdvanced()
        assert rag is not None
        assert isinstance(rag, PolicyRAG)
    
    def test_document_loading(self, sample_policy):
        """Test documents load successfully"""
        rag = PolicyRAG()
        rag.load_documents(sample_policy)
        
        assert rag.vectorstore is not None
        assert rag.qa_chain is not None
        assert rag.retriever is not None
    
    def test_load_multiple_documents(self):
        """Test loading multiple documents"""
        rag = PolicyRAG()
        docs = [
            "Policy document 1: Shipping information",
            "Policy document 2: Return information"
        ]
        rag.load_documents(docs)
        
        assert rag.vectorstore is not None


# ============================================================================
# QUESTION ANSWERING TESTS
# ============================================================================

class TestQuestionAnswering:
    """Test RAG QA functionality"""
    
    def test_ask_about_shipping(self, rag_system):
        """Test asking about shipping policy"""
        result = rag_system.ask("What are the shipping options?")
        
        assert isinstance(result, dict)
        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        
        answer = result["answer"].lower()
        # Should mention shipping options
        assert "shipping" in answer or "standard" in answer or "express" in answer
    
    def test_ask_about_returns(self, rag_system):
        """Test asking about return policy"""
        result = rag_system.ask("How long do I have to return an item?")
        
        answer = result["answer"].lower()
        # Should mention 30 days
        assert "30" in answer or "thirty" in answer
    
    def test_ask_about_free_shipping(self, rag_system):
        """Test asking about free shipping"""
        result = rag_system.ask("Do you offer free shipping?")
        
        answer = result["answer"].lower()
        # Should mention $50 threshold
        assert "50" in answer or "free" in answer
    
    def test_out_of_scope_question(self, rag_system):
        """Test handling of out-of-scope questions"""
        result = rag_system.ask("What is the weather today?")
        
        answer = result["answer"].lower()
        # Should say they don't have that information
        assert (
            "don't have" in answer or
            "not in" in answer or
            "cannot" in answer or
            "unable" in answer
        )
    
    def test_ask_with_sources(self, rag_system):
        """Test formatted answer with sources"""
        result = rag_system.ask_with_sources("What is your return policy?")
        
        assert isinstance(result, str)
        assert "Question:" in result
        assert "Answer:" in result
        assert "Sources:" in result
    
    def test_source_documents_returned(self, rag_system):
        """Test that source documents are returned"""
        result = rag_system.ask("What shipping methods are available?")
        
        sources = result["source_documents"]
        assert len(sources) > 0
        assert all(hasattr(doc, "page_content") for doc in sources)
    
    def test_batch_questions(self, rag_system):
        """Test batch question processing"""
        questions = [
            "What is the shipping cost?",
            "Can I return items?",
            "How do I contact support?"
        ]
        
        results = rag_system.batch_questions(questions)
        
        assert len(results) == 3
        assert all("answer" in r for r in results)


# ============================================================================
# ADVANCED RAG TESTS
# ============================================================================

class TestAdvancedRAG:
    """Test advanced RAG features"""
    
    def test_confidence_assessment(self, rag_advanced):
        """Test answer with confidence assessment"""
        result = rag_advanced.ask_with_confidence("What is the return window?")
        
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence_assessment" in result
        
        assessment = result["confidence_assessment"]
        assert len(assessment) > 0
        # Should mention confidence level
        assert (
            "HIGH" in assessment.upper() or
            "MEDIUM" in assessment.upper() or
            "LOW" in assessment.upper()
        )
    
    def test_similarity_search(self, rag_advanced):
        """Test semantic similarity search"""
        results = rag_advanced.search_similar("shipping information", k=3)
        
        assert len(results) <= 3
        assert all(len(r) == 2 for r in results)  # (doc, score) tuples
        
        # Check first result
        doc, score = results[0]
        assert hasattr(doc, "page_content")
        assert isinstance(score, float)
    
    def test_search_returns_relevant_content(self, rag_advanced):
        """Test search returns relevant content"""
        results = rag_advanced.search_similar("returns and refunds", k=5)
        
        # At least one result should mention returns or refunds
        content = " ".join(doc.page_content.lower() for doc, _ in results)
        assert "return" in content or "refund" in content


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestVariousQuestions:
    """Test various question types"""
    
    @pytest.mark.parametrize("question,expected_keywords", [
        (
            "How much does standard shipping cost?",
            ["5.99", "$", "standard", "shipping"]
        ),
        (
            "What is the restocking fee?",
            ["15", "restocking", "fee", "percent", "%"]
        ),
        (
            "How can I contact customer service?",
            ["phone", "email", "support", "contact"]
        ),
    ])
    def test_specific_policy_questions(self, rag_system, question, expected_keywords):
        """Test specific policy questions"""
        result = rag_system.ask(question)
        answer = result["answer"].lower()
        
        # Should mention at least some of the keywords
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer)
        assert matches >= 1


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_ask_before_loading_documents(self):
        """Test error when asking before loading documents"""
        rag = PolicyRAG()
        
        with pytest.raises(ValueError) as exc_info:
            rag.ask("What is the policy?")
        
        assert "No documents loaded" in str(exc_info.value)
    
    def test_empty_question(self, rag_system):
        """Test handling of empty question"""
        result = rag_system.ask("")
        
        # Should return something without crashing
        assert result is not None
        assert "answer" in result
    
    def test_very_long_question(self, rag_system):
        """Test handling of very long question"""
        long_question = "What is the shipping policy? " * 50
        result = rag_system.ask(long_question)
        
        # Should handle without crashing
        assert result is not None
        assert "answer" in result
    
    def test_question_with_special_characters(self, rag_system):
        """Test question with special characters"""
        result = rag_system.ask("What's the cost for shipping ($)?")
        
        assert result is not None
        assert len(result["answer"]) > 0


# ============================================================================
# RETRIEVAL QUALITY TESTS
# ============================================================================

class TestRetrievalQuality:
    """Test retrieval quality"""
    
    def test_retrieves_relevant_chunks(self, rag_system):
        """Test that retrieval finds relevant content"""
        result = rag_system.ask("What is the shipping cost?")
        
        sources = result["source_documents"]
        # Combine all source content
        all_content = " ".join(doc.page_content.lower() for doc in sources)
        
        # Should contain shipping information
        assert "shipping" in all_content
    
    def test_answer_cites_sections(self, rag_system):
        """Test that answers reference policy sections"""
        result = rag_system.ask("What is your return policy?")
        answer = result["answer"]
        
        # Should have some content (not just "I don't know")
        assert len(answer) > 20


# ============================================================================
# SKIP IF NO API KEY
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip tests requiring API key if not available"""
    if not os.getenv("OPENAI_API_KEY"):
        skip_api = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            item.add_marker(skip_api)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

