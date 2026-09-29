import math
import os
from typing import Optional, Tuple
from PIL import Image

from geopy.geocoders import Nominatim

from smolagents import tool, CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool, LiteLLMModel
from smolagents.utils import encode_image_base64, make_image_url


@tool
def get_city_coordinates(city_name: str) -> str:
    """
    Returns the exact latitude and longitude for any city or location name.

    Args:
        city_name: The name (str) of the city (e.g., 'London' or 'Maranello').

    Returns:
        str: The latitude and longitude as string separated by a comma (e.g., '51.5074, -0.1278').
    """
    geolocator = Nominatim(user_agent="gotham_agent")
    location = geolocator.geocode(city_name)
    if location:
        return f"{location.latitude}, {location.longitude}"
    return "Location not found"


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)


def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = LiteLLMModel(
        model_id="ollama_chat/llama3.1:8b",
        host="http://localhost:11434",
        num_ctx=32768,
        max_tokens=8192,
        options={
            "temperature": 0.0,
            "num_predict": 4096
        }
    )
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


web_agent_model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:7b",
    host="http://localhost:11434",
    num_ctx=32768,
    max_tokens=4096,
    options={
        "temperature": 0.0,
        "num_predict": 4096
    }
)

web_agent = CodeAgent(
    model=web_agent_model,
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        get_city_coordinates,
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)

manager_agent_model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:7b",
    host="http://localhost:11434",
    num_ctx=32768,
    max_tokens=8192,
    options={
        "temperature": 0.0,
        "num_predict": 4096
    }
)

manager_agent = CodeAgent(
    model=manager_agent_model,
    tools=[get_city_coordinates, calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)

manager_agent.visualize()

manager_agent.run("""
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
""")

quit()

task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time."""


# request = task
request = f"""
<task_instructions>
Execute the following task strictly step-by-step.

TASK:
{task}
</task_instructions>

<execution_rules>
1. WORK IN SMALL, ATOMIC STEPS. You are allowed to call only ONE tool or perform ONE discrete action per step.
2. ABSOLUTELY FORBIDDEN from writing a single monolithic Python script containing long `for` loops, heavy string manipulation, `re.findall` HTML parsing, or complex `if/elif` blocks to solve the whole task at once.
3. DO NOT RUSH. If you execute a code block, it must do exactly one thing (e.g., call your custom travel time tool for a single location or extract a small, specific set of data).
4. EVALUATE EACH STEP. After receiving a tool output (Observation), stop, analyze the data you just received, update your internal state, and only then proceed to the next step.
</execution_rules>
"""

result = agent.run(request)

print(result)