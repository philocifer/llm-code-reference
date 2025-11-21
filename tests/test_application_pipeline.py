"""
Pytest Tests for Application Pipeline (Real API Calls)
=======================================================
Tests OrderProcessor - integration of multiple LLM techniques.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from application_pipeline import (
    OrderProcessor,
    ExtractedOrder,
    PolicyCheck,
    FulfillmentDecision,
    ProcessingResult,
    SAMPLE_POLICY
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def processor():
    """Create processor with sample policy"""
    return OrderProcessor(SAMPLE_POLICY)


@pytest.fixture
def simple_order_text():
    """Simple order text"""
    return """
    Order from John Smith (john@email.com)
    
    I'd like to order 5 wireless keyboards at $30 each.
    
    Ship to:
    123 Main Street
    Austin, TX 78701
    """


@pytest.fixture
def complex_order_text():
    """Complex order with special requests"""
    return """
    Order Request
    
    Customer: Maria Rodriguez (maria.r@company.com)
    Product: Ergonomic Office Chairs
    Quantity: 15 units
    Price: $250 per unit
    
    Shipping Address:
    TechCorp Inc.
    456 Business Park Drive, Suite 200
    San Francisco, CA 94105
    
    Special Instructions:
    Please coordinate delivery with building management.
    Need white glove delivery service.
    """


@pytest.fixture
def large_quantity_order():
    """Order with large quantity"""
    return """
    Need to order 80 standing desks for new office build-out.
    Customer: Sarah Chen
    Email: sarah@startup.com
    Price: $400 each
    Ship to: 789 Innovation Way, Seattle, WA 98101
    This is for a business account.
    """


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Test processor initialization"""
    
    def test_processor_creation(self):
        """Test processor creates successfully"""
        processor = OrderProcessor(SAMPLE_POLICY)
        assert processor is not None
        assert processor.llm is not None
        assert processor.policy_vectorstore is not None
    
    def test_processor_with_custom_policy(self):
        """Test processor with custom policy"""
        custom_policy = "Custom policy text: Maximum 50 items per order."
        processor = OrderProcessor(custom_policy)
        assert processor is not None


# ============================================================================
# STEP 1: EXTRACTION TESTS
# ============================================================================

class TestOrderExtraction:
    """Test order extraction step"""
    
    def test_extract_simple_order(self, processor, simple_order_text):
        """Test extraction of simple order"""
        result = processor.extract_order(simple_order_text)
        
        assert isinstance(result, ExtractedOrder)
        assert result.customer_name is not None
        assert "smith" in result.customer_name.lower() or "john" in result.customer_name.lower()
        assert result.email is not None
        assert "@" in result.email
        assert result.product_name is not None
        assert "keyboard" in result.product_name.lower()
        assert result.quantity == 5
        assert result.shipping_address is not None
        assert "austin" in result.shipping_address.lower()
    
    def test_extract_complex_order(self, processor, complex_order_text):
        """Test extraction of complex order"""
        result = processor.extract_order(complex_order_text)
        
        assert isinstance(result, ExtractedOrder)
        assert result.customer_name is not None
        assert result.quantity == 15
        assert result.unit_price is not None
        assert result.special_instructions is not None
    
    def test_extract_with_missing_email(self, processor):
        """Test extraction when email is missing"""
        order_text = """
        Customer: Jane Doe
        Product: Mouse
        Quantity: 2
        Ship to: 100 Oak St, Portland OR
        """
        
        result = processor.extract_order(order_text)
        
        assert isinstance(result, ExtractedOrder)
        assert result.customer_name is not None
        # Email may be None
        assert result.email is None or isinstance(result.email, str)


# ============================================================================
# STEP 2: POLICY CHECK TESTS
# ============================================================================

class TestPolicyCheck:
    """Test policy compliance checking"""
    
    def test_check_compliant_order(self, processor, simple_order_text):
        """Test policy check for compliant order"""
        order = processor.extract_order(simple_order_text)
        policy_check = processor.check_policy_compliance(order)
        
        assert isinstance(policy_check, PolicyCheck)
        assert isinstance(policy_check.is_compliant, bool)
        assert 0 <= policy_check.compliance_score <= 100
        assert isinstance(policy_check.passed_checks, list)
        assert isinstance(policy_check.failed_checks, list)
        assert isinstance(policy_check.warnings, list)
    
    def test_check_large_quantity_order(self, processor, large_quantity_order):
        """Test policy check flags large quantities"""
        order = processor.extract_order(large_quantity_order)
        policy_check = processor.check_policy_compliance(order)
        
        assert isinstance(policy_check, PolicyCheck)
        # Large quantity (80) should trigger warnings or failures
        # According to SAMPLE_POLICY, max is 100 but over 50 needs verification
        all_text = (
            " ".join(policy_check.failed_checks + policy_check.warnings)
        ).lower()
        
        # Should mention quantity, verification, or business
        assert (
            "quantity" in all_text or
            "verification" in all_text or
            "business" in all_text or
            "50" in all_text or
            policy_check.compliance_score < 100
        )
    
    def test_policy_check_uses_rag(self, processor, simple_order_text):
        """Test that policy check uses RAG system"""
        order = processor.extract_order(simple_order_text)
        policy_check = processor.check_policy_compliance(order)
        
        # Should produce meaningful checks
        total_checks = len(policy_check.passed_checks) + len(policy_check.failed_checks)
        assert total_checks > 0


# ============================================================================
# STEP 3: DECISION TESTS
# ============================================================================

class TestFulfillmentDecision:
    """Test fulfillment decision step"""
    
    def test_assess_and_decide_simple_order(self, processor, simple_order_text):
        """Test decision for simple order"""
        order = processor.extract_order(simple_order_text)
        policy_check = processor.check_policy_compliance(order)
        decision = processor.assess_and_decide(order, policy_check)
        
        assert isinstance(decision, FulfillmentDecision)
        assert 0 <= decision.fulfillment_score <= 100
        assert decision.decision in [
            "APPROVE", "APPROVE_CONDITIONAL", "HOLD", "DECLINE"
        ]
        assert len(decision.reasoning) > 0
        assert isinstance(decision.actions, list)
    
    def test_decision_considers_policy(self, processor, large_quantity_order):
        """Test that decision considers policy compliance"""
        order = processor.extract_order(large_quantity_order)
        policy_check = processor.check_policy_compliance(order)
        decision = processor.assess_and_decide(order, policy_check)
        
        # Large order should have actions or conditions
        reasoning = decision.reasoning.lower()
        
        # Should mention verification, quantity, or business
        assert any(keyword in reasoning for keyword in [
            "quantity", "verification", "verify", "business", "large"
        ])


# ============================================================================
# END-TO-END TESTS
# ============================================================================

class TestEndToEnd:
    """Test complete processing pipeline"""
    
    def test_process_complete_order(self, processor, simple_order_text):
        """Test processing order end-to-end"""
        result = processor.process_order(simple_order_text)
        
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.order, ExtractedOrder)
        assert isinstance(result.policy_check, PolicyCheck)
        assert 0 <= result.fulfillment_score <= 100
        assert result.decision in [
            "APPROVE", "APPROVE_CONDITIONAL", "HOLD", "DECLINE"
        ]
        assert len(result.reasoning) > 0
    
    def test_generate_report(self, processor, simple_order_text):
        """Test report generation"""
        result = processor.process_order(simple_order_text)
        report = processor.generate_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert "ORDER PROCESSING REPORT" in report
        assert "ORDER INFORMATION" in report
        assert "POLICY COMPLIANCE" in report
        assert "FULFILLMENT ASSESSMENT" in report
        assert "FINAL DECISION" in report
    
    def test_report_contains_order_details(self, processor, simple_order_text):
        """Test that report contains order details"""
        result = processor.process_order(simple_order_text)
        report = processor.generate_report(result)
        
        # Should contain customer name
        assert result.order.customer_name in report
        # Should contain product
        assert result.order.product_name in report
    
    def test_process_multiple_orders(self, processor, simple_order_text, complex_order_text):
        """Test processing multiple orders"""
        result1 = processor.process_order(simple_order_text)
        result2 = processor.process_order(complex_order_text)
        
        # Both should succeed
        assert isinstance(result1, ProcessingResult)
        assert isinstance(result2, ProcessingResult)
        assert result1.decision in [
            "APPROVE", "APPROVE_CONDITIONAL", "HOLD", "DECLINE"
        ]
        assert result2.decision in [
            "APPROVE", "APPROVE_CONDITIONAL", "HOLD", "DECLINE"
        ]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test integration of different components"""
    
    def test_extraction_feeds_policy_check(self, processor, complex_order_text):
        """Test that extraction results flow to policy check"""
        result = processor.process_order(complex_order_text)
        
        # Extracted data should be reflected in policy check
        order = result.order
        policy = result.policy_check
        
        # Policy check should consider the extracted quantity
        assert order.quantity > 0
        assert policy.compliance_score >= 0
    
    def test_policy_check_affects_decision(self, processor, large_quantity_order):
        """Test that policy compliance affects decision"""
        result = processor.process_order(large_quantity_order)
        
        # Low compliance should influence decision
        if result.policy_check.compliance_score < 70:
            # Decision should reflect concerns
            assert (
                result.decision in ["HOLD", "APPROVE_CONDITIONAL"] or
                len(result.actions) > 0
            )
    
    def test_end_to_end_data_flow(self, processor, simple_order_text):
        """Test data flows through entire pipeline"""
        result = processor.process_order(simple_order_text)
        
        # Order data should flow through
        assert result.order.customer_name is not None
        
        # Policy check should have run
        assert result.policy_check.compliance_score >= 0
        
        # Decision should consider everything
        assert len(result.reasoning) > 50  # Substantial reasoning


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestVariousOrderScenarios:
    """Test various order scenarios"""
    
    @pytest.mark.parametrize("quantity,should_flag", [
        (5, False),     # Normal quantity
        (25, False),    # Medium quantity
        (60, True),     # Over 50 - should need verification
        (90, True),     # Large quantity
    ])
    def test_quantity_thresholds(self, processor, quantity, should_flag):
        """Test handling of different quantities"""
        order_text = f"""
        Customer: Test Customer
        Email: test@example.com
        Product: Office Chairs
        Quantity: {quantity}
        Price: $100 each
        Ship to: 123 Test St, City, State
        """
        
        result = processor.process_order(order_text)
        
        if should_flag:
            # Should have lower compliance or require actions
            assert (
                result.policy_check.compliance_score < 100 or
                len(result.actions) > 0 or
                result.decision in ["HOLD", "APPROVE_CONDITIONAL"]
            )


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""
    
    def test_minimal_order_text(self, processor):
        """Test with minimal information"""
        minimal = "Customer: John. Product: Chair. Quantity: 1. Address: 123 St."
        
        result = processor.process_order(minimal)
        
        # Should handle without crashing
        assert isinstance(result, ProcessingResult)
    
    def test_order_with_special_characters(self, processor):
        """Test order with special characters"""
        order_text = """
        Customer: José García
        Email: jose@company.com
        Product: Desk (36" x 72")
        Quantity: 3
        Price: $450 each
        Ship to: 123 Main St. #5, Austin, TX
        """
        
        result = processor.process_order(order_text)
        
        # Should handle special characters
        assert isinstance(result, ProcessingResult)
    
    def test_order_with_missing_price(self, processor):
        """Test order without price information"""
        order_text = """
        Customer: Alice Smith
        Product: Monitor
        Quantity: 2
        Ship to: 456 Oak Ave, Portland OR
        """
        
        result = processor.process_order(order_text)
        
        # Should handle missing price
        assert isinstance(result, ProcessingResult)


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

