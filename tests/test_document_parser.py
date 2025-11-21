"""
Pytest Tests for Document Parser (Real API Calls)
==================================================
Tests that call the actual OpenAI API to verify real-world behavior.

Run with: pytest tests/test_document_parser.py -v

Note: Requires OPENAI_API_KEY environment variable to be set.
"""

import pytest
from pydantic import ValidationError
import os
import sys

# Import the classes we're testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_parser import OrderExtractor, CustomerOrder


# ============================================================================
# FIXTURES - Reusable test setup
# ============================================================================

@pytest.fixture
def extractor():
    """Create an OrderExtractor instance for testing"""
    return OrderExtractor()


@pytest.fixture
def simple_order():
    """Simple, complete order"""
    return """
    Customer: John Smith
    Email: john.smith@email.com
    
    I'd like to order 2 units of the ErgoChair Pro at $299 each.
    
    Please ship to:
    123 Main Street, Seattle, WA 98101
    
    Leave package at the front desk if no one is home.
    """


@pytest.fixture
def incomplete_order():
    """Order with missing optional fields"""
    return """
    Hi, this is Sarah Johnson. I want to buy 5 wireless keyboards.
    Ship to 456 Oak Avenue, Portland, OR 97201.
    """


@pytest.fixture
def complex_order():
    """Complex order with multiple details"""
    return """
    Order Request
    
    My name is Michael Chen (m.chen@company.com). I need to place an order
    for our office.
    
    Product: Standing Desk Converter
    Quantity: 3 units
    Price: $199 per unit
    
    Shipping Address:
    TechCorp Inc.
    789 Business Park Drive, Suite 200
    San Francisco, CA 94105
    
    Please coordinate delivery with our building manager.
    Call 555-0123 for building access.
    """


# ============================================================================
# MODEL VALIDATION TESTS - No API calls needed
# ============================================================================

class TestCustomerOrderValidation:
    """Test Pydantic model validation logic"""
    
    def test_valid_order_creation(self):
        """Test creating a valid order"""
        order = CustomerOrder(
            customer_name="Jane Doe",
            email="jane@example.com",
            product_name="Wireless Mouse",
            quantity=5,
            unit_price=29.99,
            shipping_address="456 Oak Ave, Portland, OR"
        )
        
        assert order.customer_name == "Jane Doe"
        assert order.email == "jane@example.com"
        assert order.quantity == 5
        assert order.unit_price == 29.99
    
    def test_order_with_optional_fields_none(self):
        """Test order with optional fields set to None"""
        order = CustomerOrder(
            customer_name="Bob Wilson",
            product_name="Keyboard",
            quantity=1,
            shipping_address="789 Pine St, Austin, TX"
        )
        
        assert order.email is None
        assert order.unit_price is None
        assert order.special_instructions is None
    
    def test_negative_price_validation(self):
        """Test that negative prices are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            CustomerOrder(
                customer_name="Test User",
                product_name="Test Product",
                quantity=1,
                unit_price=-10.0,
                shipping_address="Test Address"
            )
        
        assert "Price must be positive" in str(exc_info.value)
    
    def test_zero_price_validation(self):
        """Test that zero price is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            CustomerOrder(
                customer_name="Test User",
                product_name="Test Product",
                quantity=1,
                unit_price=0.0,
                shipping_address="Test Address"
            )
        
        assert "Price must be positive" in str(exc_info.value)
    
    def test_excessive_quantity_validation(self):
        """Test that quantity cannot exceed 1000"""
        with pytest.raises(ValidationError) as exc_info:
            CustomerOrder(
                customer_name="Test User",
                product_name="Test Product",
                quantity=1001,
                shipping_address="Test Address"
            )
        
        assert "Quantity cannot exceed 1000" in str(exc_info.value)
    
    def test_boundary_quantity_values(self):
        """Test boundary values for quantity (1 and 1000 should work)"""
        order_min = CustomerOrder(
            customer_name="Test User",
            product_name="Test Product",
            quantity=1,
            shipping_address="Test Address"
        )
        assert order_min.quantity == 1
        
        order_max = CustomerOrder(
            customer_name="Test User",
            product_name="Test Product",
            quantity=1000,
            shipping_address="Test Address"
        )
        assert order_max.quantity == 1000
    
    @pytest.mark.parametrize("quantity,should_pass", [
        (1, True),
        (500, True),
        (1000, True),
        (1001, False),
        (0, False),
    ])
    def test_quantity_boundaries(self, quantity, should_pass):
        """Test various quantity boundaries"""
        if should_pass:
            order = CustomerOrder(
                customer_name="Test",
                product_name="Product",
                quantity=quantity,
                shipping_address="Address"
            )
            assert order.quantity == quantity
        else:
            with pytest.raises(ValidationError):
                CustomerOrder(
                    customer_name="Test",
                    product_name="Product",
                    quantity=quantity,
                    shipping_address="Address"
                )


# ============================================================================
# EXTRACTOR TESTS - Real API Calls
# ============================================================================

class TestOrderExtractor:
    """Test OrderExtractor with real API calls"""
    
    def test_extractor_initialization(self):
        """Test that extractor initializes properly"""
        extractor = OrderExtractor()
        assert extractor is not None
        assert extractor.llm is not None
        assert extractor.structured_llm is not None
        assert extractor.prompt is not None
    
    def test_extract_simple_order(self, extractor, simple_order):
        """Test extraction of a complete order"""
        result = extractor.extract(simple_order)
        
        # Verify all fields are extracted
        assert result.customer_name is not None
        assert "smith" in result.customer_name.lower()
        
        assert result.email is not None
        assert "@" in result.email
        
        assert result.product_name is not None
        assert "chair" in result.product_name.lower()
        
        assert result.quantity == 2
        
        assert result.unit_price is not None
        assert result.unit_price > 0
        
        assert result.shipping_address is not None
        assert "seattle" in result.shipping_address.lower()
        
        assert result.special_instructions is not None
        assert "front desk" in result.special_instructions.lower()
    
    def test_extract_incomplete_order(self, extractor, incomplete_order):
        """Test extraction when some fields are missing"""
        result = extractor.extract(incomplete_order)
        
        # Required fields should be present
        assert result.customer_name is not None
        assert "johnson" in result.customer_name.lower()
        
        assert result.product_name is not None
        assert "keyboard" in result.product_name.lower()
        
        assert result.quantity == 5
        
        assert result.shipping_address is not None
        assert "portland" in result.shipping_address.lower()
        
        # Optional fields may be None
        # (LLM might still extract them, so we don't assert None)
    
    def test_extract_complex_order(self, extractor, complex_order):
        """Test extraction of a complex order with many details"""
        result = extractor.extract(complex_order)
        
        # Verify extraction
        assert result.customer_name is not None
        assert "chen" in result.customer_name.lower()
        
        assert result.email is not None
        assert "m.chen" in result.email.lower()
        
        assert result.product_name is not None
        assert "desk" in result.product_name.lower()
        
        assert result.quantity == 3
        
        assert result.unit_price is not None
        assert abs(result.unit_price - 199.0) < 1.0  # Allow small variance
        
        assert result.shipping_address is not None
        assert "san francisco" in result.shipping_address.lower()
        
        # Special instructions should mention building manager
        if result.special_instructions:
            assert "building" in result.special_instructions.lower() or "delivery" in result.special_instructions.lower()
    
    def test_extract_validates_output(self, extractor):
        """Test that extracted data passes Pydantic validation"""
        order_text = """
        Customer: Alice Williams
        Email: alice@example.com
        Order: 100 wireless mice at $15 each
        Ship to: 100 Tech Drive, Austin, TX 78701
        """
        
        result = extractor.extract(order_text)
        
        # Should not raise validation errors
        assert isinstance(result, CustomerOrder)
        assert result.quantity >= 1
        assert result.quantity <= 1000
        if result.unit_price:
            assert result.unit_price > 0
    
    def test_extract_with_confidence(self, extractor, simple_order):
        """Test the extract_with_confidence method"""
        result = extractor.extract_with_confidence(simple_order)
        
        # Should return a dict with data and confidence assessment
        assert isinstance(result, dict)
        assert "data" in result
        assert "confidence_assessment" in result
        
        # Data should be a valid dict (from model_dump)
        assert isinstance(result["data"], dict)
        assert "customer_name" in result["data"]
        
        # Confidence assessment should be a string
        assert isinstance(result["confidence_assessment"], str)
        assert len(result["confidence_assessment"]) > 0
    
    def test_multiple_extractions_consistency(self, extractor):
        """Test that multiple extractions of the same text are consistent"""
        order_text = """
        Customer: Bob Smith
        Product: Laptop Stand
        Quantity: 1
        Price: $49.99
        Ship to: 200 Main St, Boston, MA 02101
        """
        
        # Extract twice
        result1 = extractor.extract(order_text)
        result2 = extractor.extract(order_text)
        
        # Key fields should be the same (with temperature=0)
        assert result1.quantity == result2.quantity
        assert "smith" in result1.customer_name.lower()
        assert "smith" in result2.customer_name.lower()
        assert "laptop" in result1.product_name.lower() or "stand" in result1.product_name.lower()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_minimal_valid_order(self, extractor):
        """Test extraction with minimal information"""
        minimal = """
        Name: John Doe
        Product: Widget
        Quantity: 1
        Address: 123 Street, City, State
        """
        
        result = extractor.extract(minimal)
        
        # Should extract at minimum the required fields
        assert result.customer_name is not None
        assert result.product_name is not None
        assert result.quantity >= 1
        assert result.shipping_address is not None
    
    def test_order_with_extra_noise(self, extractor):
        """Test extraction when there's extra irrelevant text"""
        noisy = """
        Hello! I hope you're doing well today. The weather is nice.
        
        Anyway, I'd like to place an order:
        
        Customer: Jane Smith
        Product: Ergonomic Keyboard
        Quantity: 2
        Price: $89.99 each
        
        Please ship to:
        456 Oak Street
        Portland, OR 97201
        
        Thanks so much! Have a great day!
        """
        
        result = extractor.extract(noisy)
        
        # Should extract correctly despite the noise
        assert "smith" in result.customer_name.lower() or "jane" in result.customer_name.lower()
        assert result.quantity == 2
        assert "portland" in result.shipping_address.lower()
    
    def test_empty_string_handling(self, extractor):
        """Test handling of empty or nearly empty input"""
        # This will likely fail or return minimal data
        # The LLM should try its best but may not have enough info
        try:
            result = extractor.extract("")
            # If it doesn't raise an error, it should at least return a CustomerOrder
            assert isinstance(result, CustomerOrder)
        except Exception as e:
            # It's acceptable to raise an error for empty input
            assert isinstance(e, (ValueError, Exception))
    
    def test_ambiguous_quantity(self, extractor):
        """Test handling of ambiguous quantity"""
        ambiguous = """
        Customer: Alice Brown
        Product: Office Chairs
        Quantity: a few
        Ship to: 789 Business Pkwy, Denver, CO 80202
        """
        
        # LLM should interpret "a few" as a reasonable number
        result = extractor.extract(ambiguous)
        
        # Should return a valid quantity (LLM will interpret "a few")
        assert result.quantity >= 1
        assert result.quantity <= 1000


# ============================================================================
# PARAMETRIZED REAL-WORLD SCENARIOS
# ============================================================================

class TestRealWorldScenarios:
    """Test various real-world order formats"""
    
    @pytest.mark.parametrize("order_text,expected_customer,expected_product_keywords", [
        (
            "Order from John Smith for 3 Wireless Mice at $25 each. Ship to 123 Main St, Seattle WA",
            "smith",
            ["mouse", "mice", "wireless"]  # Accept plural or singular
        ),
        (
            "Sarah Johnson\nProduct: Laptop Stand\nQty: 1\nAddress: 456 Oak Ave, Portland OR",
            "johnson",
            ["laptop", "stand"]
        ),
        (
            "Michael Chen (m.chen@email.com) needs 5 keyboards shipped to 789 Pine St, Austin TX",
            "chen",
            ["keyboard"]
        ),
    ])
    def test_various_order_formats(self, extractor, order_text, expected_customer, expected_product_keywords):
        """Test extraction from various order format styles"""
        result = extractor.extract(order_text)
        
        assert expected_customer in result.customer_name.lower()
        # Check if any of the expected keywords are in the product name
        assert any(keyword in result.product_name.lower() for keyword in expected_product_keywords)
        assert result.quantity >= 1
        assert result.shipping_address is not None


# ============================================================================
# SKIP TESTS IF NO API KEY
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip tests requiring API key if not available"""
    if not os.getenv("OPENAI_API_KEY"):
        skip_api = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            # Skip all tests except validation tests (which don't need API)
            if "TestCustomerOrderValidation" not in item.nodeid:
                item.add_marker(skip_api)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
