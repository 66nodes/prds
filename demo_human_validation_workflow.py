#!/usr/bin/env python3
"""
Interactive Demo of Human-in-the-Loop Validation Workflow
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import uuid

class ValidationPrompt:
    """Simplified validation prompt for demo"""
    
    def __init__(self, type: str, question: str, context: str, options=None, timeout=None):
        self.id = str(uuid.uuid4())
        self.type = type
        self.question = question
        self.context = context
        self.options = options or []
        self.timeout = timeout
        self.created_at = datetime.now()

class HumanValidationDemo:
    """Demo of the human validation workflow"""
    
    def __init__(self):
        self.active_validations = {}
        self.validation_history = []
        self.conversation_context = {
            "project": "AI-Powered Strategic Planning Platform",
            "phase": "PRD Generation", 
            "user": "Product Manager",
            "ai_confidence": 0.85
        }
    
    def display_context(self):
        """Display current conversation context"""
        print("🏢 Strategic Planning Platform - Human Validation Demo")
        print("=" * 60)
        print(f"📋 Project: {self.conversation_context['project']}")
        print(f"🔄 Phase: {self.conversation_context['phase']}")
        print(f"👤 User: {self.conversation_context['user']}")
        print(f"🤖 AI Confidence: {self.conversation_context['ai_confidence']*100:.1f}%")
        print("=" * 60)
    
    def create_validation_prompt(self, type: str, question: str, context: str, **kwargs):
        """Create a new validation prompt"""
        prompt = ValidationPrompt(type, question, context, **kwargs)
        self.active_validations[prompt.id] = prompt
        return prompt
    
    def display_prompt(self, prompt: ValidationPrompt):
        """Display a validation prompt to the user"""
        print(f"\n⚠️  HUMAN INPUT REQUIRED")
        print("-" * 40)
        print(f"🔢 Validation ID: {prompt.id[:8]}...")
        print(f"📝 Type: {prompt.type.upper()}")
        print(f"❓ Question: {prompt.question}")
        print(f"📄 Context: {prompt.context}")
        
        if prompt.options:
            print(f"📋 Options:")
            for i, option in enumerate(prompt.options, 1):
                if isinstance(option, dict):
                    print(f"   {i}. {option['label']} ({option['value']})")
                    if option.get('description'):
                        print(f"      → {option['description']}")
                else:
                    print(f"   {i}. {option}")
        
        if prompt.timeout:
            print(f"⏱️  Timeout: {prompt.timeout/1000:.0f} seconds")
        
        print("-" * 40)
    
    def simulate_user_response(self, prompt: ValidationPrompt, response_type: str = "approve"):
        """Simulate user response for demo purposes"""
        responses = {
            "approve": {
                "approved": True,
                "feedback": "This looks great! I approve this approach.",
                "response": {"decision": "approved"}
            },
            "reject": {
                "approved": False,
                "feedback": "This needs revision. Please consider alternative approaches.",
                "response": {"decision": "rejected", "reason": "complexity_concerns"}
            },
            "choice": {
                "approved": True,
                "feedback": "Selected the best option for our use case.",
                "response": {"choice": prompt.options[1]["value"] if prompt.options else "default"}
            },
            "input": {
                "approved": True,
                "feedback": "Added detailed requirements.",
                "response": {"input": "Additional security requirements: MFA, audit logging, data encryption at rest"}
            }
        }
        
        return responses.get(response_type, responses["approve"])
    
    def process_response(self, validation_id: str, user_response: Dict[str, Any]):
        """Process user response and update system state"""
        if validation_id not in self.active_validations:
            print(f"❌ Validation {validation_id[:8]} not found!")
            return False
        
        prompt = self.active_validations[validation_id]
        
        # Record in history
        history_entry = {
            "id": validation_id,
            "prompt": {
                "type": prompt.type,
                "question": prompt.question,
                "context": prompt.context
            },
            "response": user_response,
            "timestamp": datetime.now(),
            "approved": user_response.get("approved", False)
        }
        
        self.validation_history.append(history_entry)
        
        # Remove from active
        del self.active_validations[validation_id]
        
        # Display response
        print(f"\n✅ VALIDATION RESPONSE PROCESSED")
        print("-" * 40)
        print(f"🔢 ID: {validation_id[:8]}...")
        print(f"👤 Decision: {'✅ APPROVED' if user_response['approved'] else '❌ REJECTED'}")
        print(f"💬 Feedback: {user_response.get('feedback', 'No feedback provided')}")
        
        if user_response['approved']:
            print(f"🚀 AI will continue with approved approach")
            # Update AI confidence based on approval
            self.conversation_context['ai_confidence'] = min(0.95, self.conversation_context['ai_confidence'] + 0.05)
        else:
            print(f"🔄 AI will revise approach based on feedback") 
            # Decrease AI confidence on rejection
            self.conversation_context['ai_confidence'] = max(0.60, self.conversation_context['ai_confidence'] - 0.10)
        
        print(f"🤖 Updated AI Confidence: {self.conversation_context['ai_confidence']*100:.1f}%")
        print("-" * 40)
        
        return True
    
    def display_validation_history(self):
        """Display validation history"""
        if not self.validation_history:
            print("\n📊 No validation history available")
            return
        
        print(f"\n📊 VALIDATION HISTORY ({len(self.validation_history)} items)")
        print("=" * 60)
        
        for entry in self.validation_history:
            status = "✅ APPROVED" if entry["approved"] else "❌ REJECTED"
            print(f"🔢 {entry['id'][:8]}... | {entry['prompt']['type'].upper()} | {status}")
            print(f"   ❓ {entry['prompt']['question'][:50]}...")
            print(f"   💬 {entry['response'].get('feedback', 'No feedback')[:50]}...")
            print(f"   ⏰ {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()

async def run_demo():
    """Run the interactive demo"""
    demo = HumanValidationDemo()
    
    demo.display_context()
    
    print("\n🚀 Starting Human Validation Workflow Demo...")
    print("This demo simulates the human-in-the-loop validation system")
    print("that we've implemented for the Strategic Planning Platform.\n")
    
    # Scenario 1: Architecture Approval
    print("🎬 SCENARIO 1: Architecture Approval")
    print("The AI has proposed a system architecture and needs approval...")
    
    arch_prompt = demo.create_validation_prompt(
        type="approval",
        question="Do you approve the proposed microservices architecture?",
        context="The AI suggests implementing a microservices architecture with API Gateway, Authentication Service, PRD Generation Service, GraphRAG Validation, and Human Validation workflows. This will provide scalability but increases deployment complexity.",
        timeout=30000
    )
    
    demo.display_prompt(arch_prompt)
    
    print("\n👤 Product Manager is reviewing...")
    await asyncio.sleep(1)  # Simulate thinking time
    
    user_response = demo.simulate_user_response(arch_prompt, "approve")
    demo.process_response(arch_prompt.id, user_response)
    
    # Scenario 2: Technology Choice
    print("\n\n🎬 SCENARIO 2: Technology Stack Choice")
    print("The AI needs to choose between database technologies...")
    
    tech_prompt = demo.create_validation_prompt(
        type="choice",
        question="Which database combination should we use for GraphRAG?",
        context="The system needs to store both structured data and perform vector similarity search for hallucination detection. The AI has identified three viable options.",
        options=[
            {
                "label": "PostgreSQL + pgvector",
                "value": "postgres_pgvector",
                "description": "Traditional SQL with vector extension, familiar but limited vector performance"
            },
            {
                "label": "Neo4j + Milvus",
                "value": "neo4j_milvus",
                "description": "Graph database + vector database, optimal for GraphRAG but more complex"
            },
            {
                "label": "MongoDB + Pinecone",
                "value": "mongodb_pinecone", 
                "description": "Document database + managed vector service, good balance of performance and simplicity"
            }
        ]
    )
    
    demo.display_prompt(tech_prompt)
    
    print("\n👤 Product Manager is evaluating options...")
    await asyncio.sleep(1)
    
    choice_response = demo.simulate_user_response(tech_prompt, "choice")
    demo.process_response(tech_prompt.id, choice_response)
    
    # Scenario 3: Feature Scope Input
    print("\n\n🎬 SCENARIO 3: Additional Requirements Input")
    print("The AI needs more details about security requirements...")
    
    input_prompt = demo.create_validation_prompt(
        type="input",
        question="Please specify additional security requirements",
        context="The AI has generated basic security requirements (authentication, authorization, data validation) but needs domain-specific security constraints for this enterprise application."
    )
    
    demo.display_prompt(input_prompt)
    
    print("\n👤 Product Manager is adding requirements...")
    await asyncio.sleep(1)
    
    input_response = demo.simulate_user_response(input_prompt, "input")
    demo.process_response(input_prompt.id, input_response)
    
    # Scenario 4: Complex Feature Rejection
    print("\n\n🎬 SCENARIO 4: Feature Complexity Rejection")
    print("The AI proposes an advanced feature that may be too complex...")
    
    complex_prompt = demo.create_validation_prompt(
        type="approval",
        question="Should we implement real-time collaborative editing with conflict resolution?",
        context="The AI suggests adding Google Docs-style real-time collaborative editing to the PRD interface. This would require WebSocket connections, operational transforms, conflict resolution algorithms, and significantly more development time."
    )
    
    demo.display_prompt(complex_prompt)
    
    print("\n👤 Product Manager is considering complexity vs. value...")
    await asyncio.sleep(1)
    
    reject_response = demo.simulate_user_response(complex_prompt, "reject")
    demo.process_response(complex_prompt.id, reject_response)
    
    # Display final summary
    demo.display_validation_history()
    
    print(f"\n🎯 DEMO SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully demonstrated Human-in-the-Loop validation")
    print(f"📊 Processed {len(demo.validation_history)} validation requests")
    print(f"🤖 Final AI Confidence: {demo.conversation_context['ai_confidence']*100:.1f}%")
    print(f"🧠 The system learns from human feedback to improve future decisions")
    
    approved_count = sum(1 for h in demo.validation_history if h["approved"])
    rejected_count = len(demo.validation_history) - approved_count
    print(f"📈 Approval Rate: {approved_count}/{len(demo.validation_history)} ({approved_count/len(demo.validation_history)*100:.1f}%)")
    
    print("\n🚀 Human Validation System is working perfectly!")
    print("This system ensures human oversight of critical AI decisions")
    print("while maintaining development velocity and system intelligence.")

if __name__ == "__main__":
    asyncio.run(run_demo())