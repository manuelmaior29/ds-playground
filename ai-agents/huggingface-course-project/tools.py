import random
from langchain.tools import Tool

def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns a summary.
    Useful for factual questions or when you need information on a specific topic.
    """
    try:
        return f"Simulated Wikipedia search result for '{query}': Information about {query} is vast and covers many aspects, from its historical context to its modern-day applications."
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {e}"

def get_weather_info(location: str) -> str:
    """
    Fetches dummy weather information for a given location.
    """
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Windy", "temp_c": 18},
        {"condition": "Clear", "temp_c": 24},
    ]
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']} deg. Celsius."

search_wikipedia_tool = Tool(
    name="search_wikipedia",
    func=search_wikipedia,
    description="Searches Wikipedia for the given query and returns a summary. Useful for factual questions or when you need information on a specific topic."
)

weather_tool = Tool(
    name="get_weather_info",
    func= get_weather_info,
    description="Fetches dummy weather information for a given location."
)