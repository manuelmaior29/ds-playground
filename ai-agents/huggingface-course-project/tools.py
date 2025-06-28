from langchain.tools import tool

from langchain_community.document_loaders import WikipediaLoader

@tool
def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns a summary.
    Useful for factual questions or when you need information on a specific topic.
    """
    try:
        result = WikipediaLoader(query=query, load_max_docs=2).load()
        result = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in result
            ]
        )
        return result
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {e}"

@tool
def get_unreversed_text(text: str) -> str:
    """
    Unreverses a reversed text.
    """
    return text[::-1]

@tool
### File processing ###
def download_file(url: str) -> str:
    """
    Downloads a file from the given URL and returns the file path.
    """
    pass



