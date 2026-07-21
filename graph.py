
from langgraph.graph import StateGraph, START, END
from state import TravelState
from agents import (
    run_planner_agent, run_flight_agent, run_hotel_agent,
    run_weather_agent, run_budget_agent, run_attraction_agent,
    run_itinerary_agent, run_travel_tips_agent, run_response_generator
)

def evaluate_extraction_sanity(state: TravelState) -> str:
    """Evaluates early parsing errors to safely redirect or short-circuit faulty queries."""
    if state.get("errors") or not state.get("destination"):
        return "short_circuit_exit"
    return "proceed_logistics_pipeline"

def construct_graph() -> StateGraph:
    builder = StateGraph(TravelState)
    
    # 1. Register operational graph nodes
    builder.add_node("Planner", run_planner_agent)
    builder.add_node("FlightExplorer", run_flight_agent)
    builder.add_node("HotelExplorer", run_hotel_agent)
    builder.add_node("WeatherTracker", run_weather_agent)
    builder.add_node("BudgetInspector", run_budget_agent)
    builder.add_node("AttractionFinder", run_attraction_agent)
    builder.add_node("ItineraryArchitect", run_itinerary_agent)
    builder.add_node("SurvivalTipsSpecialist", run_travel_tips_agent)
    builder.add_node("DossierCompositor", run_response_generator)
    
    # 2. Wire entry transition
    builder.add_edge(START, "Planner")
    
    # 3. Conditional routing:
    # Route either to DossierCompositor on error, or to FlightExplorer to start parallel flow
    builder.add_conditional_edges(
    "Planner",
    evaluate_extraction_sanity,
    {
        "short_circuit_exit": "DossierCompositor",
        "proceed_logistics_pipeline": "FlightExplorer"  # Must be a single node string name
    }
)
    
    # 4. Fan-out / Parallel execution:
    # Planner -> FlightExplorer & HotelExplorer run in parallel
    builder.add_edge("Planner", "HotelExplorer")
    
    # Downstream execution paths
    builder.add_edge("FlightExplorer", "WeatherTracker")
    builder.add_edge("HotelExplorer", "BudgetInspector")
    
    # Convergence point: Both parallel paths feed into AttractionFinder
    builder.add_edge("WeatherTracker", "AttractionFinder")
    builder.add_edge("BudgetInspector", "AttractionFinder")
    
    # Final sequential pipeline
    builder.add_edge("AttractionFinder", "ItineraryArchitect")
    builder.add_edge("ItineraryArchitect", "SurvivalTipsSpecialist")
    builder.add_edge("SurvivalTipsSpecialist", "DossierCompositor")
    builder.add_edge("DossierCompositor", END)
    
    return builder.compile()

# Instantiated usable orchestrator object
travel_planner_application = construct_graph()