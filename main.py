"""
Main driver interface script executing the compiled LangGraph architecture.
"""

import os
from dotenv import load_dotenv
from graph import travel_planner_application

# Load active environment variable flags
load_dotenv()

def prompt_user_session():
    print("=" * 70)
    print("  ⭐ Production Agentic AI Multi-Agent Travel Planner Engine ⭐ ")
    print("=" * 70)
    
    if not os.getenv("MISTRAL_API_KEY"):
        print("⚠️  Warning: MISTRAL_API_KEY env key identifier missing from environment settings.")
        print("Please check your local configuration context profile adjustments.\n")
        return

    sample_query = "Plan a 6-day trip from Delhi to Japan with a budget of 200000 INR."
    print(f"Executing standard input test flow target:\n > \"{sample_query}\"\n")
    print("Orchestrating active multi-agent execution pipeline graph processing nodes...")
    
    initial_payload = {
        "user_query": sample_query,
        "conversation_history": [],
        "errors": []
    }
    
    try:
        execution_context = travel_planner_application.invoke(initial_payload)
        
        print("\n" + "#" * 70)
        print("  GENERATED DESTINATION TRAVEL PLAN DOSSIER OUTCOME ")
        print("#" * 70 + "\n")
        
        if execution_context.get("errors"):
            print("❌ Pipeline execution encountered handling exceptions:")
            for err in execution_context["errors"]:
                print(f" - {err}")
        else:
            print(execution_context.get("final_answer"))
            
    except Exception as runtime_fault:
        print(f"💥 Critical Application Orchestration Core Fault: {str(runtime_fault)}")

if __name__ == "__main__":
    prompt_user_session()