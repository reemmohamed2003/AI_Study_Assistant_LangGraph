"""
CONCEPT: Tools
--------------

Tools available to the AI Study Assistant.

1. search_web
   Searches the web for technical documentation,
   tutorials, articles, and learning resources.

2. scrape_page
   Fetches and extracts readable content from a web page.

3. propose_preference
   Signals that the user has stated a learning preference.

   The tool itself does NOT save anything.

   graph.py intercepts this tool call and sends it
   to the Human-in-the-Loop review node.
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


HEADERS = {
    "User-Agent": "Mozilla/5.0 (LangGraph learning project)"
}


# ---------------------------------------------------------
# SEARCH WEB
# ---------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """
    Search the web for programming, AI, machine learning,
    technical documentation, tutorials, or learning resources.

    Use this before scrape_page when a URL is not available.
    """

    try:

        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=HEADERS,
            timeout=10,
        )

        resp.raise_for_status()

    except requests.RequestException as e:

        return f"Search failed: {e}"

    soup = BeautifulSoup(
        resp.text,
        "html.parser"
    )

    results = []

    for block in soup.select(
        ".result__body"
    )[:5]:

        title_tag = block.select_one(
            ".result__a"
        )

        snippet_tag = block.select_one(
            ".result__snippet"
        )

        if not title_tag:
            continue

        title = title_tag.get_text(
            strip=True
        )

        url = title_tag.get(
            "href",
            ""
        )

        snippet = (
            snippet_tag.get_text(
                strip=True
            )
            if snippet_tag
            else ""
        )

        results.append(
            f"- {title}\n"
            f"  {snippet}\n"
            f"  {url}"
        )

    if not results:

        return (
            f"No results found for '{query}'."
        )

    return "\n\n".join(results)


# ---------------------------------------------------------
# SCRAPE PAGE
# ---------------------------------------------------------

@tool
def scrape_page(url: str) -> str:
    """
    Fetch a web page and extract readable text.

    Useful for reading technical articles,
    tutorials, documentation, or learning resources.
    """

    try:

        resp = requests.get(
            url,
            headers=HEADERS,
            timeout=10,
        )

        resp.raise_for_status()

    except requests.RequestException as e:

        return (
            f"Could not fetch {url}: {e}"
        )

    soup = BeautifulSoup(
        resp.text,
        "html.parser"
    )

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
        ]
    ):
        tag.decompose()

    chunks = [
        el.get_text(
            " ",
            strip=True
        )
        for el in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "li",
                "p",
            ]
        )
    ]

    text = "\n".join(
        chunk
        for chunk in chunks
        if len(chunk) > 20
    )

    return (
        text[:3000]
        if text
        else "No readable content found on that page."
    )


# ---------------------------------------------------------
# PROPOSE PREFERENCE
# ---------------------------------------------------------

@tool
def propose_preference(
    preference: str,
    value: str,
) -> str:
    """
    Propose a learning preference for long-term memory.

    Examples:

        explanation_style -> practical examples
        difficulty_level -> beginner
        response_style -> concise

    This tool does NOT save the preference directly.

    The graph intercepts the tool call and sends it
    to the Human-in-the-Loop review node.
    """

    return "pending human review"


# ---------------------------------------------------------
# TOOL GROUPS
# ---------------------------------------------------------

SEARCH_TOOLS = [
    search_web,
    scrape_page,
]

ALL_TOOLS = [
    search_web,
    scrape_page,
    propose_preference,
]