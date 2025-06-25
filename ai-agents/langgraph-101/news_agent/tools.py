from langchain.tools import Tool
import random

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

weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)