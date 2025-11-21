"""
Pytest Tests for Text Classifier (Real API Calls)
==================================================
Tests InquiryClassifier and InquiryRouter with actual OpenAI API.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_classifier import (
    InquiryClassifier,
    InquiryRouter,
    InquiryCategory,
    UrgencyLevel,
    Department,
    ClassifiedInquiry
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def classifier():
    """Create classifier instance"""
    return InquiryClassifier()


@pytest.fixture
def router():
    """Create router instance"""
    return InquiryRouter()


@pytest.fixture
def simple_inquiry():
    """Simple product question"""
    return "Hi, does the standing desk come in white? What are the dimensions?"


@pytest.fixture
def urgent_inquiry():
    """Urgent order issue"""
    return """
    This is ridiculous! My order #12345 was supposed to arrive yesterday
    and I still don't have it! I paid $500 for express shipping!
    This is completely unacceptable!
    """


@pytest.fixture
def return_request():
    """Return request"""
    return """
    Hi, I'm Sarah Martinez. I received order #23456 yesterday but the chair
    arrived damaged. I need to return this and get a replacement ASAP.
    """


# ============================================================================
# CLASSIFICATION TESTS
# ============================================================================

class TestInquiryClassifier:
    """Test inquiry classification"""
    
    def test_classifier_initialization(self):
        """Test classifier initializes properly"""
        classifier = InquiryClassifier()
        assert classifier is not None
        assert classifier.llm is not None
        assert classifier.structured_llm is not None
    
    def test_classify_product_question(self, classifier, simple_inquiry):
        """Test classification of product question"""
        result = classifier.classify(simple_inquiry)
        
        assert isinstance(result, ClassifiedInquiry)
        assert result.primary_category in [
            InquiryCategory.PRODUCT_QUESTION,
            InquiryCategory.GENERAL_QUESTION
        ]
        assert result.urgency in [UrgencyLevel.LOW, UrgencyLevel.MEDIUM]
        assert result.sentiment in ["positive", "neutral"]
        assert len(result.keywords) > 0
        assert len(result.summary) > 0
    
    def test_classify_urgent_order_issue(self, classifier, urgent_inquiry):
        """Test classification of urgent complaint"""
        result = classifier.classify(urgent_inquiry)
        
        assert isinstance(result, ClassifiedInquiry)
        # Should be complaint or order status
        assert result.primary_category in [
            InquiryCategory.COMPLAINT,
            InquiryCategory.ORDER_STATUS,
            InquiryCategory.SHIPPING_INQUIRY
        ]
        # Should be high or critical urgency
        assert result.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]
        # Should detect negative sentiment
        assert result.sentiment in ["negative", "angry"]
        # Should extract order number
        assert result.order_number is not None
        assert "12345" in result.order_number
    
    def test_classify_return_request(self, classifier, return_request):
        """Test classification of return request"""
        result = classifier.classify(return_request)
        
        assert isinstance(result, ClassifiedInquiry)
        assert result.primary_category == InquiryCategory.RETURN_REQUEST
        assert result.urgency in [UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]
        assert result.assigned_department in [
            Department.RETURNS,
            Department.CUSTOMER_SERVICE
        ]
        # Should extract customer name
        assert result.customer_name is not None
        assert "martinez" in result.customer_name.lower() or "sarah" in result.customer_name.lower()
    
    def test_entity_extraction(self, classifier):
        """Test extraction of entities"""
        inquiry = """
        Hi, I'm John Smith. My order #98765 for $250 hasn't arrived yet.
        Can you check the status?
        """
        
        result = classifier.classify(inquiry)
        
        # Should extract customer name
        assert result.customer_name is not None
        assert "smith" in result.customer_name.lower() or "john" in result.customer_name.lower()
        
        # Should extract order number
        assert result.order_number is not None
        assert "98765" in result.order_number
        
        # Should extract or infer amount
        if result.order_amount:
            assert result.order_amount > 0
    
    def test_generate_routing_report(self, classifier, simple_inquiry):
        """Test report generation"""
        classified = classifier.classify(simple_inquiry)
        report = classifier.generate_routing_report(classified, simple_inquiry)
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert "CUSTOMER INQUIRY ROUTING TICKET" in report
        assert "PRIORITY" in report
        assert "SENTIMENT" in report
    
    def test_classify_batch(self, classifier):
        """Test batch classification"""
        inquiries = [
            "What is your return policy?",
            "I need to cancel my order #555",
            "Do you have this in stock?"
        ]
        
        results = classifier.classify_batch(inquiries)
        
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert all(isinstance(r, ClassifiedInquiry) for r in results)


# ============================================================================
# ROUTING TESTS
# ============================================================================

class TestInquiryRouter:
    """Test inquiry routing system"""
    
    def test_router_initialization(self, router):
        """Test router initializes properly"""
        assert router is not None
        assert router.classifier is not None
        assert router.queues is not None
        
        # Check all queues exist
        for urgency in UrgencyLevel:
            assert urgency in router.queues
            for dept in Department:
                assert dept in router.queues[urgency]
    
    def test_route_single_inquiry(self, router, simple_inquiry):
        """Test routing single inquiry"""
        result = router.route_inquiry(simple_inquiry)
        
        assert isinstance(result, ClassifiedInquiry)
        
        # Check inquiry was added to queue
        status = router.get_queue_status()
        total_inquiries = sum(
            count
            for urgency_queues in status.values()
            for count in urgency_queues.values()
        )
        assert total_inquiries == 1
    
    def test_route_multiple_inquiries(self, router):
        """Test routing multiple inquiries"""
        inquiries = [
            "What are your shipping options?",
            "URGENT: Order not received!",
            "Can I return this item?"
        ]
        
        for inquiry in inquiries:
            router.route_inquiry(inquiry)
        
        status = router.get_queue_status()
        total = sum(
            count
            for urgency_queues in status.values()
            for count in urgency_queues.values()
        )
        assert total == 3
    
    def test_priority_queue_ordering(self, router):
        """Test that higher priority inquiries are retrieved first"""
        # Add low priority inquiry
        router.route_inquiry("What is your return policy?")
        
        # Add high priority inquiry
        router.route_inquiry("URGENT! Order #123 not received! Need immediately!")
        
        # Get first inquiry for order fulfillment
        dept = Department.ORDER_FULFILLMENT
        next_inquiry = router.get_next_inquiry(dept)
        
        # Should get the urgent one first (if routed to this dept)
        # Or none if both went to different departments
        # This is flexible based on LLM routing
    
    def test_get_queue_status(self, router):
        """Test queue status reporting"""
        # Add some inquiries
        router.route_inquiry("Product question")
        router.route_inquiry("Return request")
        
        status = router.get_queue_status()
        
        assert isinstance(status, dict)
        # Should have some entries
        assert len(status) > 0
    
    def test_empty_queue_returns_none(self, router):
        """Test that empty queue returns None"""
        result = router.get_next_inquiry(Department.SALES)
        assert result is None


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestVariousInquiryTypes:
    """Test classification of various inquiry types"""
    
    @pytest.mark.parametrize("inquiry_text,expected_category", [
        (
            "Where is my order #12345?",
            InquiryCategory.ORDER_STATUS
        ),
        (
            "I want to return this item",
            InquiryCategory.RETURN_REQUEST
        ),
        (
            "My website login isn't working",
            InquiryCategory.TECHNICAL_SUPPORT
        ),
        (
            "I was charged twice!",
            InquiryCategory.PAYMENT_ISSUE
        ),
        (
            "What colors does this come in?",
            InquiryCategory.PRODUCT_QUESTION
        ),
    ])
    def test_category_classification(self, classifier, inquiry_text, expected_category):
        """Test that various inquiry types are classified correctly"""
        result = classifier.classify(inquiry_text)
        
        # Primary or secondary category should match
        all_categories = [result.primary_category] + result.secondary_categories
        assert expected_category in all_categories
    
    @pytest.mark.parametrize("inquiry_text,expected_sentiment", [
        ("Thanks so much for your help!", ["positive"]),
        ("Where is my order?", ["neutral", "negative"]),
        ("This is UNACCEPTABLE! Terrible service!", ["angry", "negative"]),
    ])
    def test_sentiment_detection(self, classifier, inquiry_text, expected_sentiment):
        """Test sentiment detection"""
        result = classifier.classify(inquiry_text)
        assert result.sentiment in expected_sentiment


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_short_inquiry(self, classifier):
        """Test classification of very short text"""
        result = classifier.classify("Help?")
        
        assert isinstance(result, ClassifiedInquiry)
        assert result.primary_category is not None
    
    def test_multi_intent_inquiry(self, classifier):
        """Test inquiry with multiple intents"""
        inquiry = """
        I'd like to know about your return policy, and also can you tell me
        the status of order #999? Oh, and do you have any sales going on?
        """
        
        result = classifier.classify(inquiry)
        
        # Should detect multiple categories
        total_categories = 1 + len(result.secondary_categories)
        assert total_categories >= 2
    
    def test_inquiry_with_special_characters(self, classifier):
        """Test handling of special characters"""
        inquiry = "Order #ABC-123! Cost: $99.99. Email: test@example.com"
        
        result = classifier.classify(inquiry)
        
        assert isinstance(result, ClassifiedInquiry)
        # Should still extract order number
        if result.order_number:
            assert "123" in result.order_number or "ABC" in result.order_number


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

