"""
CONCEPT: Long-term memory
-------------------------

Long-term memory stores user-specific learning preferences
that survive a single conversation and can be reused across
different threads.

Short-term memory is handled separately by LangGraph's
checkpointer and is scoped to a single thread_id.

Long-term memory is organized by:

    (user_id, "preferences")

Examples:

    explanation_style -> practical examples
    difficulty_level -> beginner
    response_style -> concise
"""

from langgraph.store.memory import InMemoryStore


# Shared long-term memory store.
store = InMemoryStore()


def get_preferences(user_id: str) -> list[dict]:
    """Read all remembered learning preferences for a user."""

    items = store.search(
        (user_id, "preferences")
    )

    return [
        item.value
        for item in items
    ]


def save_preference(
    user_id: str,
    preference: str,
    value: str,
) -> None:
    """
    Save or update a learning preference.

    The preference name is used as the memory key,
    so updating the same preference replaces the
    previous value.
    """

    preference = preference.strip().lower()
    value = value.strip()

    store.put(
        (user_id, "preferences"),
        preference,
        {
            "preference": preference,
            "value": value,
        },
    )