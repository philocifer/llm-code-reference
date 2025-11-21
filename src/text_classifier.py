"""
Text Classification and Routing System
=======================================
Classify text into categories and extract key information.

E-commerce example: Customer inquiry classification for support routing.

Features:
- Multi-class classification with confidence scoring
- Entity extraction (names, order IDs, amounts, etc.)
- Sentiment analysis
- Priority-based routing
- Batch processing support
"""

import os
from dotenv import load_dotenv
from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Classification schema
class InquiryCategory(str, Enum):
    """Types of customer inquiries"""
    ORDER_STATUS = "order_status"
    PRODUCT_QUESTION = "product_question"
    RETURN_REQUEST = "return_request"
    SHIPPING_INQUIRY = "shipping_inquiry"
    PAYMENT_ISSUE = "payment_issue"
    TECHNICAL_SUPPORT = "technical_support"
    COMPLAINT = "complaint"
    GENERAL_QUESTION = "general_question"


class UrgencyLevel(str, Enum):
    """Urgency levels for inquiries"""
    CRITICAL = "critical"  # Immediate attention needed
    HIGH = "high"  # Same day response
    MEDIUM = "medium"  # 1-2 business days
    LOW = "low"  # Standard response time


class Department(str, Enum):
    """Departments for routing"""
    SALES = "sales"
    ORDER_FULFILLMENT = "order_fulfillment"
    RETURNS = "returns"
    BILLING = "billing"
    CUSTOMER_SERVICE = "customer_service"
    TECHNICAL_SUPPORT = "technical_support"
    MANAGEMENT = "management"


class ClassifiedInquiry(BaseModel):
    """Classified and enriched customer inquiry"""
    
    # Classification
    primary_category: InquiryCategory = Field(description="Primary inquiry category")
    secondary_categories: List[InquiryCategory] = Field(
        description="Additional categories if multi-intent",
        default_factory=list
    )
    urgency: UrgencyLevel = Field(description="Urgency level")
    
    # Routing
    assigned_department: Department = Field(description="Department to handle inquiry")
    requires_manager_review: bool = Field(
        description="Whether manager review is needed",
        default=False
    )
    
    # Extracted entities
    customer_name: Optional[str] = Field(description="Customer name if mentioned", default=None)
    order_number: Optional[str] = Field(description="Order number if mentioned", default=None)
    order_amount: Optional[float] = Field(description="Order amount if mentioned", default=None)
    
    # Analysis
    sentiment: Literal["positive", "neutral", "negative", "angry"] = Field(
        description="Customer sentiment"
    )
    keywords: List[str] = Field(description="Key topics/issues mentioned")
    summary: str = Field(description="Brief summary of the inquiry")
    suggested_response_template: str = Field(
        description="Suggested response approach"
    )


class InquiryClassifier:
    """Classifier for customer inquiries"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize classifier
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for generation
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.structured_llm = self.llm.with_structured_output(ClassifiedInquiry)
    
    def classify(self, inquiry_text: str) -> ClassifiedInquiry:
        """
        Classify a customer inquiry
        
        Args:
            inquiry_text: Raw inquiry text
            
        Returns:
            Classified inquiry with routing information
        """
        prompt = ChatPromptTemplate.from_template(
            """You are a customer service AI that classifies and routes inquiries for an e-commerce company.

Analyze the following customer inquiry and classify it comprehensively.

INQUIRY:
{inquiry_text}

CLASSIFICATION GUIDELINES:

Categories:
- order_status: Questions about order tracking, delivery, status
- product_question: Product info, features, compatibility, availability
- return_request: Returns, exchanges, refunds
- shipping_inquiry: Shipping options, costs, timing, addresses
- payment_issue: Payment problems, billing, charges
- technical_support: Website issues, account access, technical problems
- complaint: Complaints, dissatisfaction, negative experiences
- general_question: General information, policies, store info

Urgency Levels:
- critical: Order not received, payment charged incorrectly, account hacked
- high: Damaged item, wrong item, urgent return, shipping problems
- medium: General order questions, product questions, standard returns
- low: Informational questions, policy questions, product browsing

Department Routing:
- sales: Product questions, availability, pre-purchase inquiries
- order_fulfillment: Order status, tracking, delivery issues
- returns: Return requests, exchanges, refund status
- billing: Payment issues, charges, invoices
- customer_service: General questions, policies, assistance
- technical_support: Website, account, technical issues
- management: Serious complaints, legal issues, escalations

Extract:
- Customer name (if mentioned)
- Order numbers (if mentioned)
- Order amounts (if mentioned)
- Sentiment (positive/neutral/negative/angry)
- Key topics/keywords
- Brief summary
- Suggested response approach

Provide your classification:"""
        )
        
        chain = prompt | self.structured_llm
        
        result = chain.invoke({
            "inquiry_text": inquiry_text
        })
        
        return result
    
    def classify_batch(self, inquiries: List[str]) -> List[ClassifiedInquiry]:
        """
        Classify multiple inquiries
        
        Args:
            inquiries: List of inquiry texts
            
        Returns:
            List of classified inquiries
        """
        results = []
        for inquiry in inquiries:
            try:
                result = self.classify(inquiry)
                results.append(result)
            except Exception as e:
                print(f"Error classifying inquiry: {e}")
                results.append(None)
        return results
    
    def generate_routing_report(self, classified: ClassifiedInquiry, original_text: str) -> str:
        """
        Generate a routing report for the classified inquiry
        
        Args:
            classified: Classified inquiry
            original_text: Original inquiry text
            
        Returns:
            Formatted routing report
        """
        report = f"""
{'='*70}
CUSTOMER INQUIRY ROUTING TICKET
{'='*70}

PRIORITY: {classified.urgency.value.upper()}
SENTIMENT: {classified.sentiment.upper()}

CLASSIFICATION
---------------------------------------------------------------------
Primary Category:     {classified.primary_category.value.replace('_', ' ').title()}
Additional Categories: {', '.join(c.value.replace('_', ' ').title() for c in classified.secondary_categories) if classified.secondary_categories else 'None'}

ROUTING
---------------------------------------------------------------------
Assigned Department:  {classified.assigned_department.value.replace('_', ' ').title()}
Manager Review:       {'⚠️  YES - REQUIRED' if classified.requires_manager_review else 'No'}

EXTRACTED INFORMATION
---------------------------------------------------------------------
Customer Name:        {classified.customer_name or 'Not mentioned'}
Order Number:         {classified.order_number or 'Not mentioned'}
Order Amount:         {f'${classified.order_amount:,.2f}' if classified.order_amount else 'Not mentioned'}

ANALYSIS
---------------------------------------------------------------------
Summary: {classified.summary}

Keywords: {', '.join(classified.keywords)}

SUGGESTED RESPONSE
---------------------------------------------------------------------
{classified.suggested_response_template}

ORIGINAL INQUIRY
---------------------------------------------------------------------
{original_text}

{'='*70}
"""
        return report


class InquiryRouter:
    """Advanced inquiry router with priority queues"""
    
    def __init__(self):
        """Initialize router with empty queues"""
        self.classifier = InquiryClassifier()
        self.queues = {
            urgency: {dept: [] for dept in Department}
            for urgency in UrgencyLevel
        }
    
    def route_inquiry(self, inquiry_text: str) -> ClassifiedInquiry:
        """
        Route an inquiry to appropriate queue
        
        Args:
            inquiry_text: Raw inquiry text
            
        Returns:
            Classified inquiry
        """
        classified = self.classifier.classify(inquiry_text)
        
        # Add to appropriate queue
        self.queues[classified.urgency][classified.assigned_department].append({
            "text": inquiry_text,
            "classified": classified,
            "timestamp": "2024-12-01 10:00:00"  # In real system, use actual timestamp
        })
        
        return classified
    
    def get_queue_status(self) -> dict:
        """
        Get status of all queues
        
        Returns:
            Dictionary with queue counts
        """
        status = {}
        for urgency in UrgencyLevel:
            status[urgency.value] = {}
            for dept in Department:
                count = len(self.queues[urgency][dept])
                if count > 0:
                    status[urgency.value][dept.value] = count
        return status
    
    def get_next_inquiry(self, department: Department) -> Optional[dict]:
        """
        Get next inquiry for a department (highest priority first)
        
        Args:
            department: Department requesting next inquiry
            
        Returns:
            Next inquiry or None if queue is empty
        """
        # Check queues in priority order
        for urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM, UrgencyLevel.LOW]:
            queue = self.queues[urgency][department]
            if queue:
                return queue.pop(0)
        return None


def main():
    """Demonstrate inquiry classification and routing"""
    
    print("="*70)
    print("Customer Inquiry Classification & Routing System")
    print("="*70)
    
    # Sample inquiries
    test_inquiries = [
        # Critical - order not received
        """
        This is ridiculous! I ordered a laptop 2 weeks ago (Order #12345) and it still
        hasn't arrived! I paid $1,200 for express shipping and now you're saying it's lost?
        I need this for work IMMEDIATELY! This is completely unacceptable!
        - James Wilson (james.w@email.com)
        """,
        
        # High - damaged item
        """
        Hi, I'm Sarah Martinez. I received my order (#23456) yesterday but the 
        ergonomic chair arrived damaged - the base is cracked. I need to return this
        and get a replacement as soon as possible. Can you help me set up a return?
        """,
        
        # Medium - product question
        """
        Hello, I'm interested in buying the standing desk converter. Does it work with
        desks that are 30 inches deep? Also, what's the weight capacity? I have dual
        monitors. Thanks!
        """,
        
        # Low - general information
        """
        Hi there, I'm just browsing your site and curious about your return policy.
        How long do I have to return items if I change my mind? Do you charge for
        return shipping?
        """,
        
        # Medium - order status
        """
        Good morning, I'm Jennifer Lee. I placed order #34567 last week and haven't
        received any shipping confirmation yet. Can you check the status? I need it
        by Friday for a gift. Order was for $150 wireless headphones.
        """,
        
        # High - payment issue + complaint
        """
        I was charged TWICE for my order (#45678)! My credit card shows two charges of
        $89.99 each but I only ordered one item. This happened last month too. 
        Someone needs to fix your payment system and refund me immediately.
        - Robert Johnson
        """,
    ]
    
    # Test 1: Basic classification
    print("\n" + "="*70)
    print("TEST 1: Individual Classification")
    print("="*70)
    
    classifier = InquiryClassifier()
    
    for i, inquiry in enumerate(test_inquiries[:2], 1):
        print(f"\n--- Inquiry #{i} ---")
        classified = classifier.classify(inquiry)
        report = classifier.generate_routing_report(classified, inquiry)
        print(report)
    
    # Test 2: Batch classification and routing
    print("\n" + "="*70)
    print("TEST 2: Batch Processing & Queue Management")
    print("="*70)
    
    router = InquiryRouter()
    
    print("\nProcessing all inquiries...")
    for inquiry in test_inquiries:
        router.route_inquiry(inquiry)
    
    print("\n✅ All inquiries processed and routed\n")
    
    # Show queue status
    print("QUEUE STATUS BY PRIORITY:")
    print("-" * 70)
    status = router.get_queue_status()
    
    for urgency in ["critical", "high", "medium", "low"]:
        if urgency in status and status[urgency]:
            print(f"\n{urgency.upper()} Priority:")
            for dept, count in status[urgency].items():
                print(f"  • {dept.replace('_', ' ').title()}: {count} inquiry(ies)")
    
    # Simulate processing queue
    print("\n" + "="*70)
    print("TEST 3: Queue Processing Simulation")
    print("="*70)
    
    # Order fulfillment department processes their queue
    print("\nOrder Fulfillment Department processing queue...")
    dept = Department.ORDER_FULFILLMENT
    
    count = 0
    while True:
        next_inquiry = router.get_next_inquiry(dept)
        if not next_inquiry:
            break
        count += 1
        classified = next_inquiry['classified']
        print(f"\n[{count}] Processing {classified.urgency.value.upper()} priority inquiry")
        print(f"    Summary: {classified.summary}")
        print(f"    Customer: {classified.customer_name or 'Unknown'}")
    
    print(f"\n✅ Order Fulfillment processed {count} inquiry(ies)")
    
    # Show updated queue status
    print("\n" + "="*70)
    print("UPDATED QUEUE STATUS:")
    print("-" * 70)
    status = router.get_queue_status()
    total_remaining = sum(
        count 
        for urgency_queues in status.values() 
        for count in urgency_queues.values()
    )
    print(f"Total inquiries remaining in all queues: {total_remaining}")


if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        main()
