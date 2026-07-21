"""
Defines the unified state schema for the LangGraph agentic workflow.
Acts as the single source of truth across all parallel and sequential nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional

class TravelState(TypedDict):
    # Core User Inputs & Extracted Parameters
    user_query: str
    source: Optional[str]
    destination: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    duration_days: Optional[int]
    travelers: Optional[int]
    budget: Optional[float]
    currency: Optional[str]

    # Agent Data Payloads
    flight_options: Optional[List[Dict[str, Any]]]
    hotel_options: Optional[List[Dict[str, Any]]]
    weather: Optional[Dict[str, Any]]
    attractions: Optional[Dict[str, Any]]
    
    # Financial Optimization Matrix
    budget_analysis: Optional[Dict[str, Any]]
    
    # Text Generation Pieces
    itinerary: Optional[str]
    travel_tips: Optional[str]
    
    # Conversational Mechanics & Memory Layer
    conversation_history: List[Dict[str, str]]
    retrieved_memories: Optional[List[str]]
    errors: List[str]
    
    # Final Structured Production Response
    final_answer: Optional[str]