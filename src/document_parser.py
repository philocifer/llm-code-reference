"""
Document Information Extraction
================================
Extract structured information from unstructured text documents.

Features:
- Extracts key fields using LLM with structured output
- Returns validated Pydantic models
- Handles missing or ambiguous information
- Confidence assessment for extracted data
"""

import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Define structured output schema
class CustomerOrder(BaseModel):
    """Structured customer order data"""
    customer_name: str = Field(description="Full name of the customer")
    email: Optional[str] = Field(description="Customer email address", default=None)
    product_name: str = Field(description="Name of the product")
    quantity: int = Field(description="Quantity ordered", ge=1)
    unit_price: Optional[float] = Field(description="Price per unit in dollars", default=None)
    shipping_address: str = Field(description="Shipping address")
    special_instructions: Optional[str] = Field(description="Any special delivery instructions", default=None)
    
    @field_validator('unit_price')
    @classmethod
    def validate_positive_price(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be at least 1')
        if v > 1000:
            raise ValueError('Quantity cannot exceed 1000')
        return v


class OrderExtractor:
    """Extract structured information from customer orders"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the extractor
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for generation (0 = deterministic)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Use structured output for more reliable extraction
        self.structured_llm = self.llm.with_structured_output(CustomerOrder)
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create extraction prompt"""
        template = """You are an order processing assistant. Extract key information from the customer order text.

Be precise and only extract information that is clearly stated. If information is missing, use null.

Order Text:
{order_text}

Extract the order information."""
        
        return ChatPromptTemplate.from_template(template)
    
    def extract(self, order_text: str) -> CustomerOrder:
        """
        Extract structured information from order text
        
        Args:
            order_text: Raw order text
            
        Returns:
            CustomerOrder object with extracted data
            
        Raises:
            ValueError: If extraction fails or validation fails
        """
        try:
            # Create the chain with structured output
            chain = self.prompt | self.structured_llm
            
            # Run extraction
            result = chain.invoke({
                "order_text": order_text
            })
            
            return result
            
        except Exception as e:
            raise ValueError(f"Extraction failed: {str(e)}")
    
    def extract_with_confidence(self, order_text: str) -> dict:
        """
        Extract information and provide confidence assessment
        
        Returns:
            Dictionary with extracted data and confidence notes
        """
        # First, get structured extraction
        extracted = self.extract(order_text)
        
        # Then assess confidence
        confidence_prompt = ChatPromptTemplate.from_template(
            """Review this order extraction. Identify any ambiguous or missing information.

Original Text: {text}

Extracted Data: {extracted}

Provide:
1. Confidence level (HIGH/MEDIUM/LOW)
2. Any concerns or ambiguities
3. Fields that need clarification

Be concise."""
        )
        
        confidence_chain = confidence_prompt | self.llm
        confidence_assessment = confidence_chain.invoke({
            "text": order_text,
            "extracted": extracted.model_dump_json()
        })
        
        return {
            "data": extracted.model_dump(),
            "confidence_assessment": confidence_assessment.content
        }


# Example usage and test cases
def main():
    """Demonstrate the order extractor"""
    
    # Sample customer orders (test data)
    sample_orders = [
        # Clear, complete order
        """
        Customer: John Smith
        Email: john.smith@email.com
        
        I'd like to order 2 units of the ErgoChair Pro at $299 each.
        
        Please ship to:
        123 Main Street
        Apt 4B
        Seattle, WA 98101
        
        Leave package at the front desk if no one is home.
        """,
        
        # Incomplete order
        """
        Hi, this is Sarah Johnson. I want to buy 5 wireless keyboards.
        Ship to 456 Oak Avenue, Portland, OR 97201.
        """,
        
        # Complex, narrative order
        """
        Order Request
        
        Hello,
        
        My name is Michael Chen (m.chen@company.com). I need to place an order
        for our office.
        
        Product: Standing Desk Converter
        Quantity: 3 units
        Price: $199 per unit
        
        Shipping Address:
        TechCorp Inc.
        789 Business Park Drive, Suite 200
        San Francisco, CA 94105
        
        Please coordinate delivery with our building manager as these are large items.
        Call 555-0123 for building access.
        
        Thank you!
        """,
    ]
    
    # Initialize extractor
    print("Initializing Order Extractor...")
    extractor = OrderExtractor()
    
    # Process each order
    for i, order in enumerate(sample_orders, 1):
        print(f"\n{'='*60}")
        print(f"Processing Order #{i}")
        print(f"{'='*60}")
        print(f"\nOriginal Text:\n{order[:200]}...")
        
        try:
            # Basic extraction
            result = extractor.extract(order)
            print(f"\n✅ Extraction Successful!")
            print(f"\nExtracted Data:")
            print(f"  Customer: {result.customer_name}")
            print(f"  Email: {result.email}" if result.email else "  Email: Not provided")
            print(f"  Product: {result.product_name}")
            print(f"  Quantity: {result.quantity}")
            print(f"  Unit Price: ${result.unit_price:,.2f}" if result.unit_price else "  Unit Price: Not provided")
            print(f"  Shipping: {result.shipping_address}")
            print(f"  Instructions: {result.special_instructions}" if result.special_instructions else "  Instructions: None")
            
            # Enhanced extraction with confidence
            print(f"\n--- Confidence Assessment ---")
            enhanced = extractor.extract_with_confidence(order)
            print(enhanced['confidence_assessment'])
            
        except ValueError as e:
            print(f"\n❌ Extraction Failed: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
    
    # Demonstrate error handling
    print(f"\n{'='*60}")
    print("Testing Error Handling")
    print(f"{'='*60}")
    
    invalid_order = "This is completely unrelated text about weather."
    try:
        result = extractor.extract(invalid_order)
        print("Extraction completed (may have null fields)")
    except Exception as e:
        print(f"❌ Handled error gracefully: {type(e).__name__}")


if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        main()

