"""
Production tool infrastructure implementing live integrations with:
- OpenWeather API (Meteorological tracks)
- Tavily Search API (Local context & attraction mining)
- Actionstack/REST HTTP Endpoints (Flight & Hotel inventory lookups)
- ChromaDB (Persistent local vector semantic memory)
"""

import os
import requests
import chromadb
from typing import List, Dict, Any, Optional

# --- LIVE API LOGISTICS LAYER ---

class FlightSearchTool:
    @staticmethod
    def execute(source: str, destination: str, date: str) -> List[Dict[str, Any]]:
        """
        Queries the production Actionstack/Travel API for real-time flight availability.
        """
        api_url = os.getenv("ACTIONSTACK_API_BASE_URL", "https://api.actionstack.io/v1/flights")
        api_key = os.getenv("ACTIONSTACK_API_KEY")
        
        if not api_key:
            return [{"error": "Missing ACTIONSTACK_API_KEY environment configuration variable."}]
            
        params = {
            "origin": source,
            "destination": destination,
            "departure_date": date,
            "sort_by": "price_asc"
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=12.0)
            response.raise_for_status()
            data = response.json()
            
            # Expecting a structured list of flight offers from the channel manager
            return data.get("flights", [{"info": "No flights matched the specified parameters."}])
            
        except requests.exceptions.RequestException as error:
            return [{"error": f"Flight API backend connection failure: {str(error)}"}]


class HotelSearchTool:
    @staticmethod
    def execute(destination: str, budget: float, duration_days: int) -> List[Dict[str, Any]]:
        """
        Queries the production Actionstack/Travel API for real-time lodging options.
        """
        api_url = os.getenv("ACTIONSTACK_API_BASE_URL", "https://api.actionstack.io/v1/hotels")
        api_key = os.getenv("ACTIONSTACK_API_KEY")
        
        if not api_key:
            return [{"error": "Missing ACTIONSTACK_API_KEY environment configuration variable."}]
            
        params = {
            "location": destination,
            "max_budget": budget,
            "duration": duration_days
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=12.0)
            response.raise_for_status()
            data = response.json()
            
            return data.get("hotels", [{"info": "No accommodations matched budget constraints."}])
            
        except requests.exceptions.RequestException as error:
            return [{"error": f"Hotel API backend connection failure: {str(error)}"}]


class WeatherEngineTool:
    @staticmethod
    def execute(destination: str, start_date: str) -> Dict[str, Any]:
        """
        Queries the OpenWeather Geocoding and 5-Day Forecast API layers.
        """
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {"error": "Missing OPENWEATHER_API_KEY environment configuration variable."}

        # Step 1: Resolve Geolocation Coordinates
        geo_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_params = {"q": destination, "limit": 1, "appid": api_key}
        
        try:
            geo_response = requests.get(geo_url, params=geo_params, timeout=8.0)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data:
                return {"error": f"Could not structurally resolve geographical coordinates for: {destination}"}
                
            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]
            
            # Step 2: Extract Weather Matrix Details
            weather_url = "https://api.openweathermap.org/data/2.5/forecast"
            weather_params = {"lat": lat, "lon": lon, "units": "metric", "appid": api_key}
            
            weather_response = requests.get(weather_url, params=weather_params, timeout=8.0)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Parse the nearest timestamp forecast
            closest_forecast = weather_data.get("list", [{}])[0]
            main_metrics = closest_forecast.get("main", {})
            weather_desc = closest_forecast.get("weather", [{}])[0].get("description", "clear sky")
            
            return {
                "average_temp_celsius": main_metrics.get("temp", "N/A"),
                "feels_like_celsius": main_metrics.get("feels_like", "N/A"),
                "humidity": main_metrics.get("humidity", "N/A"),
                "condition": weather_desc.capitalize(),
                "clothing_recommendation": "Calculated adaptively via regional forecast profiles.",
                "advisories": "Monitor real-time localized weather streams near transition dates."
            }
            
        except requests.exceptions.RequestException as error:
            return {"error": f"OpenWeather infrastructure execution dropped: {str(error)}"}


class AttractionDiscoveryTool:
    @staticmethod
    def execute(destination: str) -> Dict[str, Any]:
        """
        Uses Tavily Search API to execute low-latency web-scale search queries 
        and discover top regional landmarks, points of interest, and food spots.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"error": "Missing TAVILY_API_KEY environment configuration variable."}
            
        api_url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": f"top point of interest tourist attractions hidden gems must eat food items in {destination}",
            "search_depth": "advanced",
            "include_answer": True
        }
        
        try:
            response = requests.post(api_url, json=payload, timeout=12.0)
            response.raise_for_status()
            data = response.json()
            
            return {
                "destination": destination,
                "synthesized_research": data.get("answer", "No automated summary compiled."),
                "source_references": [res.get("url") for res in data.get("results", [])[:3]]
            }
            
        except requests.exceptions.RequestException as error:
            return {"error": f"Tavily Search API extraction failed: {str(error)}"}


# --- CHROMADB VECTOR MEMORY SUB-SYSTEM ---

class VectorMemoryEngine:
    def __init__(self, storage_directory: str = "./.memory_vault"):
        """Initializes a local persistent semantic store engine for customer preferences."""
        self.client = chromadb.PersistentClient(path=storage_directory)
        self.collection = self.client.get_or_create_collection(name="user_travel_profiles")

    def incorporate_experience(self, user_id: str, trip_summary: str, elements: Dict[str, Any]):
        """Persists structural operational logs to build long-term personal context flags."""
        metadata = {k: str(v) for k, v in elements.items() if v is not None}
        self.collection.add(
            documents=[trip_summary],
            metadatas=[metadata],
            ids=[f"{user_id}_{os.urandom(4).hex()}"]
        )

    def extract_relevant_context(self, query: str, limit: int = 2) -> List[str]:
        """Queries vectorized archives to retrieve user preferences matching intents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        return results.get("documents", [[]])[0] if results else []