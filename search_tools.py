from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities.pubmed import PubMedAPIWrapper

# Tavily Search (requires TAVILY_API_KEY)
tavily_tool = TavilySearchResults(max_results=4)

# DuckDuckGo Search (no API key needed)
duckduck_tool = DuckDuckGoSearchRun()

pubmed_wrapper = PubMedAPIWrapper(
    top_k_results=3,
    load_max_docs=3,  # limit number of papers
)
pubmed_tool = PubmedQueryRun(api_wrapper=pubmed_wrapper)
