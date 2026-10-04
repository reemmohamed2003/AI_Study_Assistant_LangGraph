"""
Streamlit interface for the AI Study Assistant.

Run:
    streamlit run app.py

The application demonstrates:

- Short-term memory
- Long-term memory
- Human-in-the-loop
- Web search tools
- User-specific learning preferences

Try:
1. Tell the assistant: "I prefer practical examples."
2. Approve saving the preference.
3. Start a new conversation.
4. Ask another technical question.
5. The assistant will use the remembered preference.
"""

import uuid
import streamlit as st
from langgraph.types import Command

from graph import graph
from memory import get_preferences
from util import extract_text


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓"
)

st.title("🎓 AI Study Assistant")

st.caption(
    "A LangGraph demo: short-term memory · long-term memory · "
    "human-in-the-loop · tools"
)


# ---------------------------------------------------------
# THREAD MANAGEMENT
# ---------------------------------------------------------

def new_thread():
    """
    Start a new conversation.

    This resets the short-term conversation memory
    displayed by the Streamlit interface.

    Long-term memory is NOT deleted.
    """

    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_interrupt = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("👤 User")

    user_id = st.text_input(
        "User id",
        value=st.session_state.get("user_id", "alice")
    )

    # If the user changes, start a new conversation
    if user_id != st.session_state.get("user_id"):

        st.session_state.user_id = user_id
        new_thread()

    # Create first thread
    if "thread_id" not in st.session_state:
        new_thread()

    # New conversation button
    if st.button(
        "🔄 New conversation (same user)",
        use_container_width=True
    ):
        new_thread()

    st.caption(
        f"thread_id: `{st.session_state.thread_id[:8]}…`"
    )

    st.caption(
        "↳ Short-term memory resets. "
        "Long-term memory stays."
    )

    # -----------------------------------------------------
    # LONG-TERM MEMORY
    # -----------------------------------------------------

    st.divider()

    st.subheader("🧠 Long-term memory")

    st.caption(
        f"What I remember about **{user_id}**, "
        "across all conversations:"
    )

    prefs = get_preferences(user_id)

    if prefs:

        for p in prefs:

            st.write(
                f"📚 **{p['preference']}**: "
                f"{p['value']}"
            )

    else:

        st.caption(
            "_Nothing saved yet — tell me how you prefer to learn!_"
        )


# ---------------------------------------------------------
# LANGGRAPH CONFIG
# ---------------------------------------------------------

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ---------------------------------------------------------
# DISPLAY AI RESPONSE
# ---------------------------------------------------------

def show_ai_reply(result: dict):

    reply = extract_text(
        result["messages"][-1].content
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

    with st.chat_message("assistant"):
        st.markdown(reply)


# ---------------------------------------------------------
# HUMAN-IN-THE-LOOP APPROVAL UI
# ---------------------------------------------------------

if st.session_state.get("pending_interrupt"):

    payload = st.session_state.pending_interrupt

    with st.chat_message("assistant"):

        st.warning(
            f"🤔 **Approval needed:** "
            f"{payload['message']}"
        )

        col1, col2 = st.columns(2)

        approve = col1.button(
            "✅ Approve & remember",
            use_container_width=True
        )

        reject = col2.button(
            "❌ Reject",
            use_container_width=True
        )

    # -----------------------------------------------------
    # RESUME GRAPH AFTER HUMAN DECISION
    # -----------------------------------------------------

    if approve or reject:

        result = graph.invoke(
            Command(
                resume={
                    "approved": approve
                }
            ),
            config=config
        )

        st.session_state.pending_interrupt = None

        # If another interrupt exists
        if "__interrupt__" in result:

            st.session_state.pending_interrupt = (
                result["__interrupt__"][0].value
            )

        else:

            show_ai_reply(result)

        st.rerun()


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

else:

    user_input = st.chat_input(
        "Ask about AI, programming, ML, or tell me how you prefer to learn..."
    )

    if user_input:

        # Add user message to UI
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # -------------------------------------------------
        # RUN LANGGRAPH
        # -------------------------------------------------

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {
                    "messages": [
                        ("user", user_input)
                    ],
                    "user_id": user_id,
                    "search_results": []
                },
                config=config
            )

        # -------------------------------------------------
        # HUMAN-IN-THE-LOOP INTERRUPT
        # -------------------------------------------------

        if "__interrupt__" in result:

            st.session_state.pending_interrupt = (
                result["__interrupt__"][0].value
            )

            st.rerun()

        else:

            show_ai_reply(result)