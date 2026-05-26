from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """ Search the web """
    return "results.."


response = search_web.invoke("Ai news")
print(response)
