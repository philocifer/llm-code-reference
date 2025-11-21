"""
End-to-End Processing Pipeline
===============================
Demonstrates integration of multiple LLM techniques in a single workflow.

E-commerce example: Complete order processing from text to fulfillment.

Combines:
- Document extraction with structured output
- RAG-based policy validation
- Multi-step analysis and decision making
- Report generation

Shows how to compose different LLM patterns into a cohesive system.
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Reuse models from other examples
class ExtractedOrder(BaseModel):
    """Extracted order data"""
    customer_name: str
    email: Optional[str] = None
    product_name: str
    quantity: int
    unit_price: Optional[float] = None
    shipping_address: str
    special_instructions: Optional[str] = None


class PolicyCheck(BaseModel):
    """Policy compliance check"""
    is_compliant: bool = Field(description="Whether order meets policy requirements")
    passed_checks: List[str] = Field(description="Policy requirements that passed")
    failed_checks: List[str] = Field(description="Policy requirements that failed")
    warnings: List[str] = Field(description="Potential issues to review")
    compliance_score: int = Field(description="Compliance score 0-100", ge=0, le=100)


class FulfillmentDecision(BaseModel):
    """Fulfillment decision data"""
    fulfillment_score: int = Field(description="Fulfillment score 0-100", ge=0, le=100)
    decision: str = Field(description="Decision: APPROVE, APPROVE_CONDITIONAL, HOLD, or DECLINE")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    actions: List[str] = Field(description="Required actions to take", default_factory=list)


class ProcessingResult(BaseModel):
    """Complete processing result"""
    order: ExtractedOrder
    policy_check: PolicyCheck
    fulfillment_score: int = Field(ge=0, le=100)
    decision: str
    reasoning: str
    actions: List[str] = Field(default_factory=list)


class OrderProcessor:
    """
    End-to-end order processor
    Demonstrates integration of multiple LLM techniques
    """
    
    def __init__(self, policy_documents: str):
        """
        Initialize processor with policy documents
        
        Args:
            policy_documents: Policy text for RAG system
        """
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize RAG for policy checks
        self._setup_policy_rag(policy_documents)
    
    def _setup_policy_rag(self, policy_documents: str):
        """Setup RAG system for policy checking"""
        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = [Document(page_content=policy_documents)]
        chunks = splitter.split_documents(docs)
        
        # Create vector store
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.policy_vectorstore = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            location=":memory:",
            collection_name="policies"
        )
    
    def extract_order(self, raw_text: str) -> ExtractedOrder:
        """
        Step 1: Extract structured data from raw order text
        
        Args:
            raw_text: Unstructured order text
            
        Returns:
            Extracted order data
        """
        structured_llm = self.llm.with_structured_output(ExtractedOrder)
        
        prompt = ChatPromptTemplate.from_template(
            """Extract order information from the following text.

Order Text:
{text}

Extract the order information:"""
        )
        
        chain = prompt | structured_llm
        
        return chain.invoke({
            "text": raw_text
        })
    
    def check_policy_compliance(self, order: ExtractedOrder) -> PolicyCheck:
        """
        Step 2: Check order against policies using RAG
        
        Args:
            order: Extracted order
            
        Returns:
            Policy compliance check
        """
        # Query relevant policies
        queries = [
            f"shipping policy for {order.quantity} units",
            f"return policy for {order.product_name}",
            "order processing requirements"
        ]
        
        # Gather relevant policy context
        relevant_policies = []
        for query in queries:
            docs = self.policy_vectorstore.similarity_search(query, k=2)
            relevant_policies.extend([doc.page_content for doc in docs])
        
        policy_context = "\n\n".join(set(relevant_policies))
        
        # Check compliance
        structured_llm = self.llm.with_structured_output(PolicyCheck)
        
        prompt = ChatPromptTemplate.from_template(
            """Review this order against our store policies.

Order:
- Customer: {customer_name}
- Email: {email}
- Product: {product_name}
- Quantity: {quantity}
- Unit Price: ${unit_price}
- Shipping: {shipping_address}
- Instructions: {special_instructions}

Relevant Policies:
{policies}

Check compliance with:
1. Quantity limits (typically 1-100 units)
2. Shipping address requirements
3. Product availability
4. Special request feasibility
5. Customer information completeness

Provide compliance check:"""
        )
        
        chain = prompt | structured_llm
        
        return chain.invoke({
            "customer_name": order.customer_name,
            "email": order.email or "Not provided",
            "product_name": order.product_name,
            "quantity": order.quantity,
            "unit_price": order.unit_price or 0,
            "shipping_address": order.shipping_address,
            "special_instructions": order.special_instructions or "None",
            "policies": policy_context
        })
    
    def assess_and_decide(
        self,
        order: ExtractedOrder,
        policy_check: PolicyCheck
    ) -> FulfillmentDecision:
        """
        Step 3: Assess feasibility and make fulfillment decision
        
        Args:
            order: Extracted order
            policy_check: Policy compliance results
            
        Returns:
            Fulfillment assessment and decision
        """
        structured_llm = self.llm.with_structured_output(FulfillmentDecision)
        
        prompt = ChatPromptTemplate.from_template(
            """Make a fulfillment decision based on the order and policy check.

Order:
- Customer: {customer_name}
- Product: {product_name}
- Quantity: {quantity}
- Value: ${order_value}

Policy Compliance:
- Compliant: {is_compliant}
- Score: {compliance_score}/100
- Passed: {passed_checks}
- Failed: {failed_checks}
- Warnings: {warnings}

Provide:
1. Fulfillment score (0-100, higher is better)
2. Decision (APPROVE / APPROVE_CONDITIONAL / HOLD / DECLINE)
3. Detailed reasoning
4. Required actions (e.g., "Verify shipping address", "Contact customer for clarification")
"""
        )
        
        chain = prompt | structured_llm
        
        order_value = (order.unit_price or 0) * order.quantity
        
        return chain.invoke({
            "customer_name": order.customer_name,
            "product_name": order.product_name,
            "quantity": order.quantity,
            "order_value": order_value,
            "is_compliant": "Yes" if policy_check.is_compliant else "No",
            "compliance_score": policy_check.compliance_score,
            "passed_checks": ", ".join(policy_check.passed_checks),
            "failed_checks": ", ".join(policy_check.failed_checks) if policy_check.failed_checks else "None",
            "warnings": ", ".join(policy_check.warnings) if policy_check.warnings else "None"
        })
    
    def process_order(self, raw_order: str) -> ProcessingResult:
        """
        Process complete order end-to-end
        
        Args:
            raw_order: Raw order text
            
        Returns:
            Complete processing result
        """
        print("\n" + "="*70)
        print("PROCESSING ORDER")
        print("="*70)
        
        # Step 1: Extract
        print("\n[1/3] Extracting order data...")
        order = self.extract_order(raw_order)
        print(f"  ✓ Customer: {order.customer_name}")
        print(f"  ✓ Product: {order.product_name}")
        print(f"  ✓ Quantity: {order.quantity}")
        
        # Step 2: Policy check
        print("\n[2/3] Checking policy compliance...")
        policy_check = self.check_policy_compliance(order)
        print(f"  ✓ Compliant: {'Yes' if policy_check.is_compliant else 'No'}")
        print(f"  ✓ Compliance Score: {policy_check.compliance_score}/100")
        
        # Step 3: Decision
        print("\n[3/3] Making fulfillment decision...")
        decision_data = self.assess_and_decide(order, policy_check)
        print(f"  ✓ Fulfillment Score: {decision_data.fulfillment_score}/100")
        print(f"  ✓ Decision: {decision_data.decision}")
        
        # Combine results
        result = ProcessingResult(
            order=order,
            policy_check=policy_check,
            fulfillment_score=decision_data.fulfillment_score,
            decision=decision_data.decision,
            reasoning=decision_data.reasoning,
            actions=decision_data.actions
        )
        
        return result
    
    def generate_report(self, result: ProcessingResult) -> str:
        """Generate detailed processing report"""
        
        order = result.order
        policy = result.policy_check
        
        order_value = (order.unit_price or 0) * order.quantity
        
        report = f"""
{'='*70}
ORDER PROCESSING REPORT
{'='*70}

ORDER INFORMATION
---------------------------------------------------------------------
Customer:         {order.customer_name}
Email:            {order.email or 'Not provided'}
Product:          {order.product_name}
Quantity:         {order.quantity}
Unit Price:       ${(order.unit_price if order.unit_price else 0):,.2f}
Total Value:      ${order_value:,.2f}
Shipping:         {order.shipping_address}
Instructions:     {order.special_instructions or 'None'}

POLICY COMPLIANCE
---------------------------------------------------------------------
Overall Compliance: {'✓ PASS' if policy.is_compliant else '✗ FAIL'}
Compliance Score:   {policy.compliance_score}/100

Passed Requirements:
{chr(10).join('  ✓ ' + check for check in policy.passed_checks)}

{"Failed Requirements:" if policy.failed_checks else ""}
{chr(10).join('  ✗ ' + check for check in policy.failed_checks) if policy.failed_checks else ""}

{"Warnings:" if policy.warnings else ""}
{chr(10).join('  ⚠ ' + warning for warning in policy.warnings) if policy.warnings else ""}

FULFILLMENT ASSESSMENT
---------------------------------------------------------------------
Fulfillment Score: {result.fulfillment_score}/100

FINAL DECISION
---------------------------------------------------------------------
Decision: {result.decision}

Reasoning:
{result.reasoning}

{"Required Actions:" if result.actions else ""}
{chr(10).join('  • ' + action for action in result.actions) if result.actions else ""}

{'='*70}
END OF REPORT
{'='*70}
"""
        return report


# Sample policy for testing
SAMPLE_POLICY = """
Store Policies:
- Standard shipping: 5-7 business days
- Express shipping available for orders under 20 units
- Maximum order quantity: 100 units per order
- Quantities over 50 require business verification
- Free shipping on orders over $50
- Returns accepted within 30 days
- Custom requests evaluated case-by-case
- All orders require valid shipping address
"""


def main():
    """Demonstrate combined processing system"""
    
    # Sample orders
    test_orders = [
        """
        Order from Maria Rodriguez (maria.r@email.com)
        
        I'd like to order 8 ErgoChair Pro chairs at $299 each for our new office.
        
        Ship to:
        TechStart Inc.
        123 Innovation Drive, Suite 400
        San Francisco, CA 94105
        
        Please coordinate delivery with building management. Call 555-0100 for access.
        """,
        
        """
        Customer: John Smith
        
        Need 75 wireless keyboards ASAP for corporate event next week.
        Shipping address: 456 Business Park, Austin, TX 78701
        
        This is urgent - can you expedite?
        """
    ]
    
    # Initialize processor
    print("Initializing Order Processor...")
    processor = OrderProcessor(SAMPLE_POLICY)
    
    # Process orders
    for i, raw_order in enumerate(test_orders, 1):
        print(f"\n\n{'#'*70}")
        print(f"ORDER #{i}")
        print(f"{'#'*70}")
        
        result = processor.process_order(raw_order)
        report = processor.generate_report(result)
        print(report)
        
        if i < len(test_orders):
            input("\nPress Enter for next order...")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
    else:
        main()
