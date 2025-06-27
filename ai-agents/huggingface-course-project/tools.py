from langchain.tools import Tool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns a summary.
    Useful for factual questions or when you need information on a specific topic.
    """
    try:
        result = WikipediaAPIWrapper().run(query)
        return result
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {e}"

def get_unreversed_text(text: str) -> str:
    """
    Unreverses a reversed text.
    """
    return text[::-1]

search_wikipedia_tool = Tool(
    name="search_wikipedia",
    func=search_wikipedia,
    description="Searches Wikipedia for the given query and returns a summary. Useful for factual questions or when you need information on a specific topic."
)

text_unreverser_tool = Tool(
    name="text_unreverser",
    func= get_unreversed_text,
    description="Unreverses a reversed text."
)

