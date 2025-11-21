"""
Pytest Tests for Multi-Step Agent (Real API Calls)
===================================================
Tests OrderFulfillmentAgent with actual API.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_step_agent import (
    OrderFulfillmentAgent,
    OrderRequest,
    OrderValidation,
    FulfillmentAssessment,
    FulfillmentDecision
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def agent():
    """Create agent instance"""
    return OrderFulfillmentAgent()


@pytest.fixture
def simple_order():
    """Simple standard order"""
    return OrderRequest(
        order_id="TEST-001",
        customer_name="Jane Smith",
        customer_tier="regular",
        product_name="Office Chair",
        quantity=2,
        requested_ship_date="2024-12-15",
        special_requests="None",
        order_value=400.00
    )


@pytest.fixture
def vip_order():
    """VIP customer order"""
    return OrderRequest(
        order_id="VIP-001",
        customer_name="Robert Johnson",
        customer_tier="vip",
        product_name="Executive Desk",
        quantity=1,
        requested_ship_date="2024-12-10",
        special_requests="White glove delivery",
        order_value=2500.00
    )


@pytest.fixture
def large_order():
    """Large quantity order"""
    return OrderRequest(
        order_id="BULK-001",
        customer_name="Sarah Chen",
        customer_tier="new",
        product_name="Ergonomic Mouse",
        quantity=75,
        requested_ship_date="2024-12-20",
        special_requests="Business account",
        order_value=3750.00
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Test agent initialization"""
    
    def test_agent_creation(self):
        """Test agent creates successfully"""
        agent = OrderFulfillmentAgent()
        assert agent is not None
        assert agent.llm is not None
    
    def test_agent_with_custom_model(self):
        """Test agent with custom model"""
        agent = OrderFulfillmentAgent(model_name="gpt-4o-mini", temperature=0.1)
        assert agent is not None


# ============================================================================
# STEP 1: VALIDATION TESTS
# ============================================================================

class TestOrderValidation:
    """Test step 1: order validation"""
    
    def test_validate_simple_order(self, agent, simple_order):
        """Test validation of simple order"""
        result = agent.step1_validate_order(simple_order)
        
        assert isinstance(result, OrderValidation)
        assert isinstance(result.is_valid, bool)
        assert 1 <= result.completeness_score <= 100
        assert isinstance(result.issues, list)
        assert isinstance(result.warnings, list)
        assert len(result.customer_history_notes) > 0
    
    def test_validate_vip_order(self, agent, vip_order):
        """Test validation considers VIP status"""
        result = agent.step1_validate_order(vip_order)
        
        assert isinstance(result, OrderValidation)
        # VIP should be mentioned in notes
        notes = result.customer_history_notes.lower()
        assert "vip" in notes or "premium" in notes or "priority" in notes
    
    def test_validate_large_quantity_order(self, agent, large_order):
        """Test validation flags large quantities"""
        result = agent.step1_validate_order(large_order)
        
        assert isinstance(result, OrderValidation)
        # Large quantity should be noted
        all_text = (
            " ".join(result.issues + result.warnings + [result.customer_history_notes])
        ).lower()
        # Should mention quantity or large order
        assert "quantity" in all_text or "large" in all_text or "75" in all_text
    
    def test_validation_completeness_score(self, agent, simple_order):
        """Test that completeness score is reasonable"""
        result = agent.step1_validate_order(simple_order)
        
        # Complete order should have high score
        assert result.completeness_score >= 60


# ============================================================================
# STEP 2: FULFILLMENT ASSESSMENT TESTS
# ============================================================================

class TestFulfillmentAssessment:
    """Test step 2: fulfillment assessment"""
    
    def test_assess_simple_order(self, agent, simple_order):
        """Test assessment of simple order"""
        validation = agent.step1_validate_order(simple_order)
        assessment = agent.step2_assess_fulfillment(simple_order, validation)
        
        assert isinstance(assessment, FulfillmentAssessment)
        assert assessment.can_fulfill in ["yes", "partial", "no"]
        assert 0 <= assessment.fulfillment_score <= 100
        assert len(assessment.inventory_status) > 0
        assert len(assessment.shipping_notes) > 0
        assert isinstance(assessment.concerns, list)
        assert isinstance(assessment.alternatives, list)
    
    def test_assess_considers_validation(self, agent, simple_order):
        """Test that assessment considers validation results"""
        # Create validation with issues
        validation = agent.step1_validate_order(simple_order)
        assessment = agent.step2_assess_fulfillment(simple_order, validation)
        
        # Should produce valid assessment
        assert isinstance(assessment, FulfillmentAssessment)
        assert assessment.fulfillment_score >= 0
    
    def test_assess_large_quantity(self, agent, large_order):
        """Test assessment of large quantity order"""
        validation = agent.step1_validate_order(large_order)
        assessment = agent.step2_assess_fulfillment(large_order, validation)
        
        # Large orders should have notes about special ordering
        all_text = (
            assessment.inventory_status.lower() +
            " " + assessment.shipping_notes.lower() +
            " " + " ".join(assessment.concerns).lower()
        )
        
        # Should mention quantity, ordering, or verification
        assert any(keyword in all_text for keyword in ["quantity", "order", "large", "special", "verification"])


# ============================================================================
# STEP 3: DECISION TESTS
# ============================================================================

class TestFulfillmentDecision:
    """Test step 3: final decision"""
    
    def test_make_decision_simple_order(self, agent, simple_order):
        """Test decision making for simple order"""
        validation = agent.step1_validate_order(simple_order)
        assessment = agent.step2_assess_fulfillment(simple_order, validation)
        decision = agent.step3_make_decision(simple_order, validation, assessment)
        
        assert isinstance(decision, FulfillmentDecision)
        assert decision.decision in ["approve", "approve_partial", "hold_for_review", "decline"]
        assert isinstance(decision.approved_items, list)
        assert len(decision.estimated_ship_date) > 0
        assert isinstance(decision.special_instructions, list)
        assert len(decision.reasoning) > 0
        assert len(decision.customer_message) > 0
    
    def test_vip_gets_favorable_treatment(self, agent, vip_order):
        """Test that VIP orders get priority"""
        validation = agent.step1_validate_order(vip_order)
        assessment = agent.step2_assess_fulfillment(vip_order, validation)
        decision = agent.step3_make_decision(vip_order, validation, assessment)
        
        # VIP should likely be approved or mentioned specially
        reasoning = decision.reasoning.lower()
        message = decision.customer_message.lower()
        
        # Should mention VIP status or priority
        assert "vip" in reasoning or "priority" in reasoning or "premium" in reasoning


# ============================================================================
# END-TO-END TESTS
# ============================================================================

class TestEndToEnd:
    """Test complete order processing"""
    
    def test_process_complete_order(self, agent, simple_order):
        """Test processing order through all steps"""
        result = agent.process_order(simple_order)
        
        assert isinstance(result, dict)
        assert "order" in result
        assert "validation" in result
        assert "assessment" in result
        assert "decision" in result
        
        # Each step should have data
        assert result["validation"]["completeness_score"] >= 0
        assert result["assessment"]["fulfillment_score"] >= 0
        assert result["decision"]["decision"] in [
            "approve", "approve_partial", "hold_for_review", "decline"
        ]
    
    def test_generate_report(self, agent, simple_order):
        """Test report generation"""
        result = agent.process_order(simple_order)
        report = agent.generate_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert "ORDER FULFILLMENT DECISION REPORT" in report
        assert "VALIDATION RESULTS" in report
        assert "FULFILLMENT ASSESSMENT" in report
        assert "FINAL DECISION" in report
        
        # Should contain order details
        assert simple_order.order_id in report
        assert simple_order.customer_name in report
    
    def test_process_multiple_orders(self, agent, simple_order, vip_order):
        """Test processing multiple orders"""
        result1 = agent.process_order(simple_order)
        result2 = agent.process_order(vip_order)
        
        # Both should succeed
        assert result1 is not None
        assert result2 is not None
        assert result1["decision"]["decision"] in [
            "approve", "approve_partial", "hold_for_review", "decline"
        ]
        assert result2["decision"]["decision"] in [
            "approve", "approve_partial", "hold_for_review", "decline"
        ]


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestVariousOrderTypes:
    """Test various order types"""
    
    @pytest.mark.parametrize("tier,expected_handling", [
        ("vip", ["priority", "vip", "premium", "expedite"]),
        ("new", ["verification", "verify", "review", "new"]),
        ("regular", ["standard", "normal", "regular"]),
    ])
    def test_customer_tier_handling(self, agent, tier, expected_handling):
        """Test handling of different customer tiers"""
        order = OrderRequest(
            order_id=f"TEST-{tier}",
            customer_name="Test Customer",
            customer_tier=tier,
            product_name="Product",
            quantity=5,
            requested_ship_date="2024-12-15",
            special_requests="None",
            order_value=500.00
        )
        
        result = agent.process_order(order)
        
        # Check if tier is considered
        all_text = (
            str(result["validation"]) +
            str(result["assessment"]) +
            str(result["decision"])
        ).lower()
        
        # Should mention some relevant term for this tier
        matches = sum(1 for keyword in expected_handling if keyword in all_text)
        assert matches >= 1


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""
    
    def test_order_with_rush_date(self, agent):
        """Test order with very tight shipping deadline"""
        order = OrderRequest(
            order_id="RUSH-001",
            customer_name="Urgent Customer",
            customer_tier="regular",
            product_name="Laptop",
            quantity=1,
            requested_ship_date="2024-11-22",  # Very soon
            special_requests="URGENT - needed tomorrow!",
            order_value=1200.00
        )
        
        result = agent.process_order(order)
        
        # Should handle without crashing
        assert result is not None
        # Should mention urgency somewhere
        all_text = str(result).lower()
        assert "urgent" in all_text or "rush" in all_text or "immediate" in all_text
    
    def test_order_with_empty_special_requests(self, agent):
        """Test order with no special requests"""
        order = OrderRequest(
            order_id="SIMPLE-001",
            customer_name="Simple Customer",
            customer_tier="regular",
            product_name="Mouse",
            quantity=1,
            requested_ship_date="2024-12-30",
            special_requests="",
            order_value=25.00
        )
        
        result = agent.process_order(order)
        
        # Should handle without issues
        assert result is not None


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

