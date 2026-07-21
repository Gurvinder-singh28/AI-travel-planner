"""
Implements individual functional execution blocks (Agents) handling core tasks.
"""

import json
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import TravelState
from tools import (
    FlightSearchTool, HotelSearchTool, WeatherEngineTool, 
    AttractionDiscoveryTool, VectorMemoryEngine
)
from prompts import PLANNER_SYSTEM_PROMPT, BUDGET_ANALYSIS_PROMPT, ITINERARY_GENERATOR_PROMPT

# Centralized Mistral Model Initializer
def fetch_llm():
    return ChatMistralAI(model="mistral-large-latest", temperature=0.2)

# Global Instance of Vector Memory Engine
memory_vault = VectorMemoryEngine()

def run_planner_agent(state: TravelState) -> dict:
    """Parses user intent, reviews vector memory history, and sets core state variables."""
    llm = fetch_llm()
    
    # Retrieve contextual traces from vectorized memory blocks
    historical_traces = memory_vault.extract_relevant_context(state["user_query"])
    memory_string = "\n".join(historical_traces) if historical_traces else "No past history logged."
    
    combined_instruction = (
        f"{PLANNER_SYSTEM_PROMPT}\n\n"
        f"Retrieved historical customer preference records:\n{memory_string}"
    )
    
    response = llm.invoke([
        SystemMessage(content=combined_instruction),
        HumanMessage(content=state["user_query"])
    ])
    
    try:
        cleaned_content = response.content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
        elif cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(cleaned_content)
    except Exception as error:
        return {"errors": state.get("errors", []) + [f"Intent Parser failed to extract JSON: {str(error)}"]}
        
    return {
        "source": data.get("source"),
        "destination": data.get("destination"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "duration_days": data.get("duration_days", 5),
        "travelers": data.get("travelers", 1),
        "budget": data.get("budget"),
        "currency": data.get("currency", "USD"),
        "retrieved_memories": historical_traces
    }

def run_flight_agent(state: TravelState) -> dict:
    """Finds transportation alternatives matching timeframes."""
    if not state.get("destination"):
        return {"errors": ["Skipped flight calculations due to unknown destination context."]}
        
    options = FlightSearchTool.execute(
        source=state.get("source", "Unknown"),
        destination=state["destination"],
        date=state.get("start_date", "2026-09-01")
    )
    return {"flight_options": options}

def run_hotel_agent(state: TravelState) -> dict:
    """Finds accommodation alternatives matching budget targets."""
    if not state.get("destination"):
        return {"errors": ["Skipped hotel tracking due to unknown destination context."]}
        
    options = HotelSearchTool.execute(
        destination=state["destination"],
        budget=state.get("budget", 100000.0),
        duration_days=state.get("duration_days", 5)
    )
    return {"hotel_options": options}

def run_weather_agent(state: TravelState) -> dict:
    """Fetches meteorological metrics and structural suggestions."""
    data = WeatherEngineTool.execute(
        destination=state.get("destination", "Tokyo"),
        start_date=state.get("start_date", "2026-09-01")
    )
    return {"weather": data}

def run_budget_agent(state: TravelState) -> dict:
    """Verifies cash allocations and flags over-budget scenarios."""
    llm = fetch_llm()
    
    prompt = BUDGET_ANALYSIS_PROMPT.format(
        destination=state.get("destination", "Target"),
        budget=state.get("budget", 0.0),
        currency=state.get("currency", "USD"),
        flights=json.dumps(state.get("flight_options", [])),
        hotels=json.dumps(state.get("hotel_options", []))
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"budget_analysis": {"summary": response.content}}

def run_attraction_agent(state: TravelState) -> dict:
    """Discovers top local sightseeing recommendations."""
    data = AttractionDiscoveryTool.execute(state.get("destination", "Tokyo"))
    return {"attractions": data}

def run_itinerary_agent(state: TravelState) -> dict:
    """Generates a contextual, structured, markdown-formatted daily schedule."""
    llm = fetch_llm()
    
    logistics_context = (
        f"Flights selected: {state.get('flight_options', [{}])[0]}\n"
        f"Accommodations selected: {state.get('hotel_options', [{}])[0]}"
    )
    
    prompt = ITINERARY_GENERATOR_PROMPT.format(
        destination=state.get("destination", "Target Location"),
        duration=state.get("duration_days", 5),
        logistics=logistics_context,
        attractions=json.dumps(state.get("attractions", {}))
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"itinerary": response.content}

def run_travel_tips_agent(state: TravelState) -> dict:
    """Compiles logistical, regulatory, safety, and cultural guidance for the destination."""
    llm = fetch_llm()
    
    prompt = f"""
    Provide essential travel tips for visiting {state.get('destination', 'the destination')}. 
    Include strict sections covering:
    1. Visa & Entry Formalities
    2. Local Currency & Payment Culture
    3. Emergency Helplines & Primary Health/Safety Advice
    4. Essential Phrases & Cultural Etiquette
    5. Transit systems and navigating local transport
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"travel_tips": response.content}

def run_response_generator(state: TravelState) -> dict:
    """Assembles all intermediate outputs into a comprehensive travel plan dossier."""
    # Write the completed journey parameters to our persistent memory engine
    if state.get("destination"):
        summary_log = f"Traveled to {state['destination']} from {state.get('source')} for {state.get('duration_days')} days."
        memory_vault.incorporate_experience(
            user_id="customer_default",
            trip_summary=summary_log,
            elements={"destination": state["destination"], "budget": state.get("budget")}
        )

    constructed_dossier = (
        f"# Complete Curated Travel Dossier: {state.get('source', 'Origin')} to {state.get('destination', 'Destination')}\n\n"
        f"**Trip Duration:** {state.get('duration_days')} Days | **Group Count:** {state.get('travelers')} Traveler(s)\n"
        f"**Financial Baseline Plan Allocation:** {state.get('budget')} {state.get('currency')}\n\n"
        f"## 1. Logistics Details (Flights & Lodging Selection)\n"
        f"Top Recommended Transport Alternative: {state.get('flight_options', [{}])[0].get('carrier')} ({state.get('flight_options', [{}])[0].get('price')} {state.get('currency')})\n"
        f"Top Recommended Accommodation Alternative: {state.get('hotel_options', [{}])[0].get('name')} ({state.get('hotel_options', [{}])[0].get('price_per_night')} {state.get('currency')}/night)\n\n"
        f"## 2. Weather Advisory & Packing Guide\n"
        f"* **Forecast Parameters:** {state.get('weather', {}).get('condition')}\n"
        f"* **Recommended Attire:** {state.get('weather', {}).get('clothing_recommendation')}\n\n"
        f"## 3. Comprehensive Day-by-Day Itinerary\n"
        f"{state.get('itinerary')}\n\n"
        f"## 4. Financial Allocation Assessment\n"
        f"{state.get('budget_analysis', {}).get('summary')}\n\n"
        f"## 5. Strategic Local Survival Tips\n"
        f"{state.get('travel_tips')}\n"
    )
    
    return {"final_answer": constructed_dossier}