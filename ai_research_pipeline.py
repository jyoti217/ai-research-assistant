# =========================
# 1. Imports
# =========================
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# =========================
# 2. Tools
# =========================

@tool
def search_web(query: str) -> list:
    """Search the web and return list of URLs."""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            urls.append(r["href"])
    return urls


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch and extract text from a webpage."""
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text()[:3000]
    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# 3. Model
# =========================

def get_model():
    return init_chat_model(
        "gpt-4o-mini",
        model_provider="openai"
    )


# =========================
# 4. Pipeline
# =========================

def build_agent():
    return create_agent(
        model=get_model(),
        tools=[search_web, fetch_text_from_url],
        system_prompt = """
You are a professional AI research assistant.

Your job:
1. Search for information
2. Read relevant sources
3. Summarize clearly

Return output in STRICT JSON format:

{
  "summary": "concise explanation",
  "key_points": ["point 1", "point 2", "point 3"],
  "sources": ["url1", "url2"]
}

Rules:
- Be concise but informative
- Only include real sources
- Do NOT return anything outside JSON
"""
    )


# =========================
# 5. Runner
# =========================

def run():
    query = input("Enter research topic: ")

    summary, sources = research_pipeline(query)

    print("\n=== SUMMARY ===\n")
    print(summary)

    print("\n=== SOURCES ===")
    for s in sources:
        print("-", s)


#pipeline 
#=======
def research_pipeline(query: str):
    print("🔎 Searching...")
    urls = search_web.invoke(query)

    print("🌐 Fetching...")
    docs = []
    for url in urls[:3]:
        text = fetch_text_from_url.invoke(url)
        docs.append((url, text))

    print("🧠 Summarizing...")
    combined_text = "\n\n".join([d[1] for d in docs])
    model = get_model()

    result = model.invoke(
        f"""
Summarize the following research:

{combined_text}

Return:
- summary
- key points
- sources
"""
    )

    return result.content, [u for u, _ in docs]


# =========================
# 6. Entry point
# =========================

if __name__ == "__main__":
    run()