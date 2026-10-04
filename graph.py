"""
CONCEPT: Putting it all together — the graph
---------------------------------------------

Flow:

        START
          |
          v
       [agent] <---------------------------+
          |                                |
   (route_after_agent)                     |
      /        \             \             |
     v          v              v            |
   END       [tools]     [human_review] ----+
              |                             ^
              +-----------------------------+

- agent:
  Calls the LLM with tools and known long-term learning preferences
  injected into the system prompt.

- tools:
  Runs search_web / scrape_page using LangGraph's ToolNode.

- human_review:
  Reached when the model calls propose_preference.
  The graph pauses using interrupt() and waits for human approval.

Short-term memory:
    MemorySaver()
    -> scoped to thread_id

Long-term memory:
    InMemoryStore()
    -> scoped to user_id
"""

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from config import llm
from state import State
from tools import ALL_TOOLS, SEARCH_TOOLS
from memory import get_preferences, save_preference, store


# ---------------------------------------------------------
# LLM + TOOLS
# ---------------------------------------------------------

llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly AI Study Assistant that helps users
learn programming, artificial intelligence, machine learning, and related
technical topics.

- Use `search_web` to find up-to-date technical documentation, tutorials,
  articles, or learning resources when needed.

- If the user clearly states a learning preference, such as preferring
  practical examples, concise explanations, visual explanations, or
  step-by-step explanations, call `propose_preference`.

- Call `propose_preference` ALONE, not combined with other tool calls
  in the same turn.

- Keep explanations clear, practical, and appropriate to the user's level.

What you already remember about this user:
{memories}
"""


# ---------------------------------------------------------
# FORMAT LONG-TERM MEMORIES
# ---------------------------------------------------------

def _format_memories(prefs: list[dict]) -> str:

    if not prefs:
        return "(nothing yet)"

    return "\n".join(
        f"- {p['preference']}: {p['value']}"
        for p in prefs
    )


# ---------------------------------------------------------
# AGENT NODE
# ---------------------------------------------------------

def agent_node(state: State) -> dict:

    # Retrieve long-term memory
    prefs = get_preferences(state["user_id"])

    # Inject memories into system prompt
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(
            memories=_format_memories(prefs)
        )
    )

    # Ask the LLM
    response = llm_with_tools.invoke(
        [system] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------

def route_after_agent(state: State) -> str:

    last = state["messages"][-1]

    # Normal answer -> END
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END

    # Preference detected -> Human-in-the-loop
    if any(
        call["name"] == "propose_preference"
        for call in last.tool_calls
    ):
        return "human_review"

    # Normal tool call -> ToolNode
    return "tools"


# ---------------------------------------------------------
# HUMAN-IN-THE-LOOP NODE
# ---------------------------------------------------------

def human_review_node(
    state: State,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:

    last = state["messages"][-1]

    # Get the proposed preference tool call
    call = next(
        c
        for c in last.tool_calls
        if c["name"] == "propose_preference"
    )

    preference = call["args"]["preference"]
    value = call["args"]["value"]

    # -----------------------------------------------------
    # HUMAN-IN-THE-LOOP
    # -----------------------------------------------------

    decision = interrupt(
        {
            "type": "preference_approval",
            "preference": preference,
            "value": value,
            "message": (
                f"Save this learning preference? "
                f"{preference}: {value}"
            ),
        }
    )

    # -----------------------------------------------------
    # HUMAN APPROVED
    # -----------------------------------------------------

    if decision.get("approved"):

        final_preference = decision.get(
            "preference",
            preference,
        )

        final_value = decision.get(
            "value",
            value,
        )

        save_preference(
            state["user_id"],
            final_preference,
            final_value,
        )

        note = (
            f"Saved: {final_preference} = {final_value}. "
            "I'll remember that."
        )

    # -----------------------------------------------------
    # HUMAN REJECTED
    # -----------------------------------------------------

    else:

        note = (
            "Okay, I won't save that learning preference."
        )

    # Every tool call must receive exactly one
    # ToolMessage with the same tool_call_id.
    return {
        "messages": [
            ToolMessage(
                content=note,
                tool_call_id=call["id"],
            )
        ]
    }


# ---------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------

builder = StateGraph(State)

builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tools",
    ToolNode(SEARCH_TOOLS),
)

builder.add_node(
    "human_review",
    human_review_node,
)


# START -> AGENT
builder.add_edge(
    START,
    "agent",
)


# AGENT -> END / TOOLS / HUMAN REVIEW
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "human_review": "human_review",
        END: END,
    },
)


# TOOLS -> AGENT
builder.add_edge(
    "tools",
    "agent",
)


# HUMAN REVIEW -> AGENT
builder.add_edge(
    "human_review",
    "agent",
)


# ---------------------------------------------------------
# MEMORY
# ---------------------------------------------------------

# Short-term memory:
# Conversation state is stored per thread_id.
checkpointer = MemorySaver()

# Long-term memory:
# User preferences are stored per user_id.
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)