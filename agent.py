from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# --- tool ---
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"


# --- model (THIS IS THE KEY FIX) ---
model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai"
)


# --- agent ---
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)


# --- run ---
result = agent.invoke({
    "messages": [{"role": "user", "content": "Whats the weather in Delhi how is pollution?"}]
})

print(result["messages"][-1].content)