"""
CONCEPT: State & Reducers
--------------------------
LangGraph passes a single State object between every node.

- messages:
  Short-term conversation memory using the add_messages reducer.

- search_results:
  Custom reducer that keeps unique search results.

- user_id:
  Identifies the user so long-term learning preferences
  can be stored separately for each user.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def append_unique(existing: list[str], new: list[str]) -> list[str]:
    """Append new items while avoiding duplicates."""
    existing = existing or []

    merged = list(existing)
    seen = set(existing)

    for item in new or []:
        if item not in seen:
            merged.append(item)
            seen.add(item)

    return merged


class State(TypedDict):
    # Short-term memory:
    # The conversation history for the current thread.
    messages: Annotated[list[AnyMessage], add_messages]

    # Identifies the user for long-term memory.
    user_id: str

    # Stores unique results returned by web-search tools.
    search_results: Annotated[list[str], append_unique]