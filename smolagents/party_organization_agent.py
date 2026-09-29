from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, LiteLLMModel, Tool, FinalAnswerTool


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    best_service = max(services, key=services.get)
    return best_service


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea.
    """

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }

    output_type = "string"

    def forward(self, category: str) -> str:
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        return themes.get(category, "No superhero party theme found for this category.")


@tool
def budget_estimator_tool(guests_count: int, theme_premium: bool) -> float:
    """
    Calculates the estimated cost of the party based on the number of guests.

    Args:
        guests_count: The number of people attending the party.
        theme_premium: True if the theme requires expensive decorations, False otherwise.
    """
    base_cost_per_guest = 50.0
    total = guests_count * base_cost_per_guest
    if theme_premium:
        total += 500.0
    return total


model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:3b",
    host="http://localhost:11434",
    num_ctx=8192
)

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        suggest_menu,
        catering_service_tool,
        budget_estimator_tool,
        SuperheroPartyThemeTool(),
        FinalAnswerTool()
    ],
    model=model,
    max_steps=7,
    verbosity_level=2,
)

prompt = """
Generate a comprehensive, step-by-step action plan to organize a test party in Gotham City. 
You must use your available tools to complete the following specific objectives:

1. Generate a unique party theme by calling 'superhero_party_theme_generator' with the category 'classic heroes'.
2. Determine the appropriate menu type by calling 'suggest_menu' for a 'superhero' occasion.
3. Find the best catering company in Gotham by passing a relevant query to 'catering_service_tool'.
4. Calculate the setup schedule using Python code: 
   - Decorating the venue takes 60 minutes.
   - Setting up the food buffet takes 40 minutes.
   - The party is strictly scheduled to start at 7:00 PM (19:00).
   - Use the 'datetime' and 'timedelta' modules to calculate the exact time you must begin the preparation.

Consolidate all findings—the theme description, selected menu, catering company name, and the exact start time—into a beautifully formatted, clear action plan for the butler.
"""

agent.run(prompt)

# agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['datetime'])
# agent = CodeAgent(tools=[], model=model)

# agent.run(
#     """
#     You need to prepare for the party. Here are the tasks:
#     1. Prepare the drinks - 30 minutes
#     2. Decorate the mansion - 60 minutes
#     3. Set up the menu - 45 minutes
#     4. Prepare the music and playlist - 45 minutes
#
#     If we start right now, at what time will the party be ready? Return answer in human-readable format.
#     """
# )

# print("--- RAW SYSTEM PROMPT TEMPLATE ---")
# print(agent.prompt_templates["system_prompt"])
# print("---------------------------")

# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
# agent.run("Prepare a menu for the party at the Batman cave.")