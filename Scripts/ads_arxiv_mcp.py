from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("ads-arxiv")
ADS_TOKEN = "edzbqJN5GWbFsA0dWAkyd1uiNpjsREZC5E4BKqLn"


@mcp.tool()
def search_ads(query: str, rows: int = 10) -> str:
    r = httpx.get(
        "https://api.adsabs.harvard.edu/v1/search/query",
        params={"q": query, "fl": "title,author,year,bibcode,abstract", "rows": rows},
        headers={"Authorization": f"Bearer {ADS_TOKEN}"},
    )
    return r.text


@mcp.tool()
def search_arxiv(query: str, max_results: int = 10) -> str:
    r = httpx.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": query, "max_results": max_results},
    )
    return r.text


if __name__ == "__main__":
    mcp.run()
