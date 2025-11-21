"""
Multi-Step Analysis Agent
==========================
An agent that performs sequential analysis with structured outputs.

E-commerce example: Order fulfillment workflow with multi-step validation.

Features:
- Multi-step reasoning workflow with chained LLM calls
- Structured output at each step using Pydantic
- Business logic implementation for order processing
- Detailed reporting with reasoning
- Type-safe data flow between steps
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Data models for structured workflow
class OrderValidation(BaseModel):
    """Validation results for customer order"""
    is_valid: bool = Field(description="Whether order meets basic requirements")
    completeness_score: int = Field(description="How complete the order is (1-100)", ge=1, le=100)
    issues: List[str] = Field(description="List of validation issues found")
    warnings: List[str] = Field(description="Non-blocking warnings")
    customer_history_notes: str = Field(description="Notes about customer history")


class FulfillmentAssessment(BaseModel):
    """Assessment of order fulfillment feasibility"""
    can_fulfill: Literal["yes", "partial", "no"] = Field(description="Can we fulfill this order")
    fulfillment_score: int = Field(description="Fulfillment readiness score 0-100", ge=0, le=100)
    inventory_status: str = Field(description="Current inventory status for requested items")
    shipping_notes: str = Field(description="Shipping feasibility and timing")
    concerns: List[str] = Field(description="Concerns about fulfillment")
    alternatives: List[str] = Field(description="Alternative options if full fulfillment not possible")


class FulfillmentDecision(BaseModel):
    """Final fulfillment decision with details"""
    decision: Literal["approve", "approve_partial", "hold_for_review", "decline"] = Field(
        description="Final decision"
    )
    approved_items: List[str] = Field(description="Items approved for shipment")
    estimated_ship_date: str = Field(description="Estimated shipping date")
    special_instructions: List[str] = Field(description="Special handling instructions", default_factory=list)
    reasoning: str = Field(description="Detailed explanation of decision")
    customer_message: str = Field(description="Message to send to customer")


class OrderRequest(BaseModel):
    """Complete order request data"""
    order_id: str
    customer_name: str
    customer_tier: Literal["new", "regular", "premium", "vip"]
    product_name: str
    quantity: int
    requested_ship_date: str
    special_requests: str
    order_value: float


class OrderFulfillmentAgent:
    """Agent that performs multi-step order fulfillment analysis"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the fulfillment agent
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for generation
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def step1_validate_order(self, order: OrderRequest) -> OrderValidation:
        """
        Step 1: Validate order completeness and customer standing
        
        Args:
            order: Order request data
            
        Returns:
            Order validation results
        """
        structured_llm = self.llm.with_structured_output(OrderValidation)
        
        prompt = ChatPromptTemplate.from_template(
            """You are an order validation specialist. Review this order for completeness and any issues.

Order Details:
- Order ID: {order_id}
- Customer: {customer_name} (Tier: {customer_tier})
- Product: {product_name}
- Quantity: {quantity}
- Requested Ship Date: {requested_ship_date}
- Special Requests: {special_requests}
- Order Value: ${order_value:,.2f}

Validation Criteria:
1. Is all required information present?
2. Is quantity reasonable (1-100 units typical)?
3. Is ship date realistic (need 2-3 days minimum)?
4. Are special requests feasible?
5. Customer tier considerations (VIP gets priority, new customers may need verification)

Provide validation assessment:"""
        )
        
        chain = prompt | structured_llm
        
        result = chain.invoke({
            **order.model_dump()
        })
        
        return result
    
    def step2_assess_fulfillment(
        self,
        order: OrderRequest,
        validation: OrderValidation
    ) -> FulfillmentAssessment:
        """
        Step 2: Assess fulfillment feasibility
        
        Args:
            order: Order request data
            validation: Validation results
            
        Returns:
            Fulfillment assessment
        """
        structured_llm = self.llm.with_structured_output(FulfillmentAssessment)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a fulfillment specialist. Assess if we can fulfill this order.

Order:
- Product: {product_name}
- Quantity: {quantity}
- Ship Date Requested: {requested_ship_date}
- Order Value: ${order_value:,.2f}

Validation Results:
- Valid: {is_valid}
- Completeness: {completeness_score}/100
- Issues: {issues}
- Warnings: {warnings}

Fulfillment Guidelines:
- Standard products: Usually in stock
- Quantities over 50: May require special ordering
- Rush shipping (< 3 days): Limited availability
- High-value orders (>$1000): Require verification for new customers

Consider:
1. Inventory availability (assume standard items in stock, special items may take 5-7 days)
2. Shipping timeline feasibility
3. Customer tier (VIP gets priority, premium gets expedited)
4. Any issues from validation

Provide fulfillment assessment:"""
        )
        
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "product_name": order.product_name,
            "quantity": order.quantity,
            "requested_ship_date": order.requested_ship_date,
            "order_value": order.order_value,
            "is_valid": validation.is_valid,
            "completeness_score": validation.completeness_score,
            "issues": ", ".join(validation.issues) if validation.issues else "None",
            "warnings": ", ".join(validation.warnings) if validation.warnings else "None"
        })
        
        return result
    
    def step3_make_decision(
        self,
        order: OrderRequest,
        validation: OrderValidation,
        assessment: FulfillmentAssessment
    ) -> FulfillmentDecision:
        """
        Step 3: Make final fulfillment decision
        
        Args:
            order: Order request data
            validation: Validation results
            assessment: Fulfillment assessment
            
        Returns:
            Final fulfillment decision
        """
        structured_llm = self.llm.with_structured_output(FulfillmentDecision)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a fulfillment decision officer. Make the final decision on this order.

Order: {order_id} - {customer_name} ({customer_tier})
Product: {product_name} x {quantity}
Value: ${order_value:,.2f}

Validation:
- Valid: {is_valid}
- Score: {completeness_score}/100

Fulfillment Assessment:
- Can Fulfill: {can_fulfill}
- Score: {fulfillment_score}/100
- Inventory: {inventory_status}
- Concerns: {concerns}

Decision Guidelines:
- Fulfillment score > 70 + Valid → Approve
- Fulfillment score 40-70 → Approve partial or hold
- Fulfillment score < 40 → Likely decline or hold for review
- VIP/Premium customers: More lenient, offer alternatives
- New customers: May need additional verification

Make decision:
1. Decision: approve/approve_partial/hold_for_review/decline
2. Items approved for shipment
3. Estimated ship date
4. Special handling instructions (if any)
5. Detailed reasoning
6. Customer-facing message explaining decision

Provide decision:"""
        )
        
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "order_id": order.order_id,
            "customer_name": order.customer_name,
            "customer_tier": order.customer_tier,
            "product_name": order.product_name,
            "quantity": order.quantity,
            "order_value": order.order_value,
            "is_valid": validation.is_valid,
            "completeness_score": validation.completeness_score,
            "can_fulfill": assessment.can_fulfill,
            "fulfillment_score": assessment.fulfillment_score,
            "inventory_status": assessment.inventory_status,
            "concerns": ", ".join(assessment.concerns) if assessment.concerns else "None"
        })
        
        return result
    
    def process_order(self, order: OrderRequest) -> Dict:
        """
        Process complete order through all steps
        
        Args:
            order: Complete order request
            
        Returns:
            Dictionary with all analysis steps and final decision
        """
        print(f"\n{'='*70}")
        print(f"Processing Order: {order.order_id}")
        print(f"{'='*70}")
        
        # Step 1: Validate
        print("\n[Step 1] Validating Order...")
        validation = self.step1_validate_order(order)
        print(f"  ✓ Valid: {validation.is_valid}")
        print(f"  ✓ Completeness: {validation.completeness_score}/100")
        
        # Step 2: Assess Fulfillment
        print("\n[Step 2] Assessing Fulfillment...")
        assessment = self.step2_assess_fulfillment(order, validation)
        print(f"  ✓ Can Fulfill: {assessment.can_fulfill}")
        print(f"  ✓ Fulfillment Score: {assessment.fulfillment_score}/100")
        
        # Step 3: Make Decision
        print("\n[Step 3] Making Final Decision...")
        decision = self.step3_make_decision(order, validation, assessment)
        print(f"  ✓ Decision: {decision.decision.upper()}")
        print(f"  ✓ Ship Date: {decision.estimated_ship_date}")
        
        return {
            "order": order.model_dump(),
            "validation": validation.model_dump(),
            "assessment": assessment.model_dump(),
            "decision": decision.model_dump()
        }
    
    def generate_report(self, result: Dict) -> str:
        """
        Generate human-readable report
        
        Args:
            result: Result from process_order
            
        Returns:
            Formatted report string
        """
        order = result["order"]
        val = result["validation"]
        assess = result["assessment"]
        dec = result["decision"]
        
        report = f"""
{'='*70}
ORDER FULFILLMENT DECISION REPORT
{'='*70}

ORDER INFORMATION
---------------------------------------------------------------------
Order ID:           {order['order_id']}
Customer:           {order['customer_name']} ({order['customer_tier'].title()})
Product:            {order['product_name']}
Quantity:           {order['quantity']}
Order Value:        ${order['order_value']:,.2f}
Requested Ship:     {order['requested_ship_date']}

VALIDATION RESULTS
---------------------------------------------------------------------
Valid:              {'✓ YES' if val['is_valid'] else '✗ NO'}
Completeness:       {val['completeness_score']}/100

Issues:
{chr(10).join('  ✗ ' + i for i in val['issues']) if val['issues'] else '  None'}

Warnings:
{chr(10).join('  ⚠ ' + w for w in val['warnings']) if val['warnings'] else '  None'}

FULFILLMENT ASSESSMENT
---------------------------------------------------------------------
Can Fulfill:        {assess['can_fulfill'].upper()}
Fulfillment Score:  {assess['fulfillment_score']}/100
Inventory Status:   {assess['inventory_status']}
Shipping Notes:     {assess['shipping_notes']}

Concerns:
{chr(10).join('  • ' + c for c in assess['concerns']) if assess['concerns'] else '  None'}

Alternatives:
{chr(10).join('  • ' + a for a in assess['alternatives']) if assess['alternatives'] else '  None'}

FINAL DECISION
---------------------------------------------------------------------
Decision:           {dec['decision'].upper().replace('_', ' ')}
Approved Items:     {', '.join(dec['approved_items'])}
Ship Date:          {dec['estimated_ship_date']}
"""
        
        if dec['special_instructions']:
            report += f"\nSpecial Instructions:\n"
            report += "\n".join('  • ' + i for i in dec['special_instructions'])
            report += "\n"
        
        report += f"""
Reasoning:
{dec['reasoning']}

Customer Message:
{dec['customer_message']}

{'='*70}
End of Report
{'='*70}
"""
        return report


def main():
    """Demonstrate the order fulfillment agent"""
    
    # Sample orders
    orders = [
        # Standard order - should approve
        OrderRequest(
            order_id="ORD-2024-001",
            customer_name="Sarah Johnson",
            customer_tier="premium",
            product_name="ErgoChair Pro",
            quantity=3,
            requested_ship_date="2024-12-05",
            special_requests="Please include assembly instructions",
            order_value=897.00
        ),
        
        # Challenging order - new customer, large quantity
        OrderRequest(
            order_id="ORD-2024-002",
            customer_name="Mike Chen",
            customer_tier="new",
            product_name="Standing Desk",
            quantity=25,
            requested_ship_date="2024-12-03",  # Rush
            special_requests="Need white glove delivery",
            order_value=12450.00
        ),
        
        # VIP order - should get priority
        OrderRequest(
            order_id="ORD-2024-003",
            customer_name="Jennifer Martinez",
            customer_tier="vip",
            product_name="Executive Chair Collection",
            quantity=5,
            requested_ship_date="2024-12-10",
            special_requests="Gift wrapping for corporate event",
            order_value=2495.00
        ),
    ]
    
    # Initialize agent
    print("Initializing Order Fulfillment Agent...")
    agent = OrderFulfillmentAgent()
    
    # Process each order
    for i, order in enumerate(orders, 1):
        result = agent.process_order(order)
        
        # Generate and print report
        report = agent.generate_report(result)
        print(report)
        
        if i < len(orders):
            input("\nPress Enter to process next order...")


if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        main()
