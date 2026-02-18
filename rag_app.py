from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

class QueryType(Enum):
    """Classify incoming queries"""
    CONTEXTUAL = "contextual"  # "What is maternity leave in Canada?"
    CALCULATION = "calculation"  # "Calculate overtime pay"
    CLARIFICATION = "clarification"  # "What about part-time workers?"
    FOLLOW_UP = "follow_up"  # Related to previous context
    SPECIALIST = "specialist"  # Needs accounting/expert input

@dataclass
class ConversationContext:
    """Maintain conversation state"""
    conversation_id: str
    current_subject: str  # e.g., "maternity_leave", "overtime_calculation"
    domain: str  # e.g., "labour_law", "accounting"
    entities: Dict[str, Any]  # Store extracted entities (location, worker_type, etc.)
    relevant_documents: List[str]  # Cache retrieved documents
    calculation_state: Dict[str, Any]  # Store intermediate calculation values
    last_query_type: Optional[QueryType] = None
    agent_chain: List[str] = None  # Track which agents handled this conversation
    
    def __post_init__(self):
        if self.agent_chain is None:
            self.agent_chain = []

class IntelligentRouter:
    """Smart routing system that maintains context"""
    
    def __init__(self, context_db):
        self.context_db = context_db
        self.active_contexts: Dict[str, ConversationContext] = {}
    
    def classify_query(self, query: str, context: ConversationContext) -> QueryType:
        """Classify the query type based on keywords and context"""
        query_lower = query.lower()
        
        # Calculation indicators
        calculation_keywords = ["calculate", "compute", "how much", "total", "pay", "rate", "overtime", "bonus"]
        if any(kw in query_lower for kw in calculation_keywords):
            return QueryType.CALCULATION
        
        # Clarification indicators - short questions, "what about", "for example"
        if any(phrase in query_lower for phrase in ["what about", "how about", "for example", "such as"]):
            if context.current_subject:  # Only if we have context
                return QueryType.CLARIFICATION
        
        # Follow-up indicators
        if any(word in query_lower for word in ["also", "additionally", "and", "then", "next", "after that"]):
            if context.calculation_state or context.current_subject:
                return QueryType.FOLLOW_UP
        
        # Check if it's a new subject
        if not context.current_subject:
            return QueryType.CONTEXTUAL
        
        # Specialist routing
        if any(phrase in query_lower for phrase in ["accounting", "tax", "deduction", "benefits calculation"]):
            return QueryType.SPECIALIST
        
        return QueryType.CONTEXTUAL
    
    def route_query(self, 
                   query: str, 
                   conversation_id: str,
                   user_id: str) -> Dict[str, Any]:
        """Main routing logic"""
        
        # Retrieve or create conversation context
        context = self._get_or_create_context(conversation_id, user_id)
        
        # Classify the query
        query_type = self.classify_query(query, context)
        context.last_query_type = query_type
        
        # Route to appropriate agent/handler
        routing_decision = self._make_routing_decision(query, query_type, context)
        
        return {
            "query_type": query_type,
            "target_agent": routing_decision["agent"],
            "context": context,
            "rag_params": routing_decision["rag_params"],
            "agent_instructions": routing_decision["instructions"]
        }
    
    def _make_routing_decision(self, 
                              query: str, 
                              query_type: QueryType,
                              context: ConversationContext) -> Dict[str, Any]:
        """Make specific routing decision based on query type"""
        
        if query_type == QueryType.CALCULATION:
            return {
                "agent": "accountant_agent",
                "rag_params": {
                    "subject": context.current_subject,
                    "context_entities": context.entities,
                    "previous_calculations": context.calculation_state,
                    "include_previous": True  # Reuse prior context
                },
                "instructions": f"""You are answering a calculation question about {context.current_subject}.
                Previous context: {json.dumps(context.entities)}
                Previous calculations: {json.dumps(context.calculation_state)}
                Focus on: Perform the calculation using this context without asking for repetition."""
            }
        
        elif query_type == QueryType.CLARIFICATION:
            return {
                "agent": "labour_law_specialist",
                "rag_params": {
                    "subject": context.current_subject,
                    "context_entities": context.entities,
                    "retrieval_depth": "shallow",  # Use existing docs
                    "focus_area": "edge_cases"
                },
                "instructions": f"""Answer this clarification about {context.current_subject}.
                Current context: {json.dumps(context.entities)}
                Keep the answer concise and specific to the clarification."""
            }
        
        elif query_type == QueryType.FOLLOW_UP:
            return {
                "agent": "coordinator_agent",
                "rag_params": {
                    "subject": context.current_subject,
                    "context_entities": context.entities,
                    "previous_state": context.calculation_state,
                    "chain_context": context.agent_chain
                },
                "instructions": f"""You are handling a follow-up to a previous discussion about {context.current_subject}.
                Agent chain: {' → '.join(context.agent_chain)}
                Maintain continuity and build on: {json.dumps(context.calculation_state)}"""
            }
        
        elif query_type == QueryType.SPECIALIST:
            return {
                "agent": "accounting_specialist",
                "rag_params": {
                    "subject": context.current_subject,
                    "context_entities": context.entities,
                    "specialist_mode": True
                },
                "instructions": f"""You are a specialized accounting expert for {context.current_subject}.
                Context: {json.dumps(context.entities)}
                Provide detailed accounting perspective."""
            }
        
        else:  # CONTEXTUAL
            return {
                "agent": "primary_rag_agent",
                "rag_params": {
                    "retrieval_depth": "full",
                    "extract_entities": True,
                    "setup_calculation_tracking": True
                },
                "instructions": f"""You are starting a new discussion about: {query}
                Extract key entities and prepare for calculation follow-ups."""
            }
    
    def _get_or_create_context(self, 
                               conversation_id: str,
                               user_id: str) -> ConversationContext:
        """Get existing context or create new one"""
        if conversation_id in self.active_contexts:
            return self.active_contexts[conversation_id]
        
        # Try to load from database
        stored_context = self.context_db.get(user_id, conversation_id)
        if stored_context:
            context = ConversationContext(**stored_context)
            self.active_contexts[conversation_id] = context
            return context
        
        # Create new context
        context = ConversationContext(
            conversation_id=conversation_id,
            current_subject="",
            domain="labour_law",
            entities={},
            relevant_documents=[],
            calculation_state={}
        )
        self.active_contexts[conversation_id] = context
        return context
    
    def update_context(self, 
                      conversation_id: str,
                      subject: Optional[str] = None,
                      entities: Optional[Dict] = None,
                      calculations: Optional[Dict] = None,
                      documents: Optional[List[str]] = None,
                      agent_name: Optional[str] = None):
        """Update context after agent response"""
        context = self.active_contexts.get(conversation_id)
        if not context:
            return
        
        if subject:
            context.current_subject = subject
        
        if entities:
            context.entities.update(entities)
        
        if calculations:
            context.calculation_state.update(calculations)
        
        if documents:
            context.relevant_documents = documents
        
        if agent_name:
            context.agent_chain.append(agent_name)


class ContextAwareRAGHandler:
    """RAG handler that respects routing context"""
    
    def __init__(self, rag_system, router: IntelligentRouter):
        self.rag = rag_system
        self.router = router
    
    def handle_query(self, 
                    query: str,
                    conversation_id: str,
                    user_id: str) -> Dict[str, Any]:
        """Handle a query with intelligent routing and context"""
        
        # Route the query
        routing = self.router.route_query(query, conversation_id, user_id)
        
        # Extract context-aware RAG parameters
        rag_params = routing["rag_params"]
        
        # Adjust RAG retrieval based on context
        if rag_params.get("include_previous"):
            # Reuse previous documents to maintain continuity
            retrieved_docs = self.rag.retrieve_with_context(
                query=query,
                previous_docs=routing["context"].relevant_documents,
                subject=rag_params.get("subject")
            )
        else:
            retrieved_docs = self.rag.retrieve(query, **rag_params)
        
        # Update context with retrieved documents
        self.router.update_context(
            conversation_id,
            documents=[doc.id for doc in retrieved_docs]
        )
        
        # Route to appropriate agent
        agent_response = self._call_agent(
            routing["target_agent"],
            query,
            retrieved_docs,
            routing["agent_instructions"],
            routing["context"]
        )
        
        return {
            "response": agent_response,
            "context": routing["context"],
            "routing": routing
        }
    
    def _call_agent(self, 
                   agent_name: str,
                   query: str,
                   documents: List,
                   instructions: str,
                   context: ConversationContext) -> str:
        """Call appropriate agent with instructions"""
        # This would integrate with your actual agent system
        # e.g., LangChain agents, OpenAI function calling, etc.
        
        agent_prompt = f"""{instructions}

User Query: {query}

Context Entities:
{json.dumps(context.entities, indent=2)}

Retrieved Documents:
{self._format_docs(documents)}

Please provide a comprehensive answer that builds on the context."""
        
        return agent_name  # Placeholder - replace with actual agent call


class QueryEnhancer:
    """Enhance queries with context before sending to RAG"""
    
    @staticmethod
    def enhance_query_with_context(query: str, context: ConversationContext) -> str:
        """Rewrite query to include implicit context"""
        
        if context.current_subject and query.lower().startswith(('what', 'how', 'when')):
            # Short clarification question - add context
            enhanced = f"Regarding {context.current_subject}, {query}"
        elif context.calculation_state:
            # Add prior calculation context
            enhanced = f"{query} [Previous context: {context.current_subject}, with {list(context.calculation_state.keys())}]"
        else:
            enhanced = query
        
        return enhanced
