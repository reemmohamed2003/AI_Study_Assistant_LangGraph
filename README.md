# 🎓 AI Study Assistant — LangGraph

An AI-powered study assistant built with **LangGraph** that demonstrates:

- 🧠 Short-Term Memory
- 💾 Long-Term Memory
- 👤 Human-in-the-Loop
- 🔎 Web Search Tools
- 🤖 LLM-based reasoning

The project was adapted from a memory-based chatbot into an **AI Study Assistant** designed to help users learn programming, Artificial Intelligence, Machine Learning, and related technical topics.

---

## 🎯 Project Objective

The goal of this project is to demonstrate how an AI assistant can combine different types of memory with Human-in-the-Loop interaction.

The assistant can:

- Answer technical and educational questions.
- Search the web for up-to-date learning resources.
- Remember the user's learning preferences.
- Keep conversation context within the current conversation.
- Ask the user for approval before saving a preference to long-term memory.

---

## 🏗️ Architecture

```text
                    ┌───────────────┐
                    │     User      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Streamlit   │
                    │      UI       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     Agent     │
                    │     (LLM)     │
                    └───────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        ┌─────────┐   ┌────────────┐  ┌──────────────┐
        │  Tools  │   │ Human      │  │     End      │
        │         │   │ Review     │  │              │
        └────┬────┘   └─────┬──────┘  └──────────────┘
             │              │
             ▼              ▼
       Search / Scrape   Approve / Reject
                            │
                            ▼
                    ┌───────────────┐
                    │ Long-Term     │
                    │    Memory     │
                    └───────────────┘
```

---

## 🧠 Memory Architecture

### 1. Short-Term Memory

Short-term memory is implemented using LangGraph's:

```python
MemorySaver()
```

It stores the conversation state for a specific:

```text
thread_id
```

This allows the assistant to remember previous messages within the same conversation.

When the user starts a new conversation, a new `thread_id` is created, so the previous conversation context is no longer available.

---

### 2. Long-Term Memory

Long-term memory is implemented using:

```python
InMemoryStore()
```

Learning preferences are stored using:

```text
(user_id, "preferences")
```

Example:

```text
User: I prefer practical examples.

Saved preference:
explanation_style → practical examples
```

Unlike short-term memory, this information can be reused across different conversation threads for the same user.

> Note: `InMemoryStore` is an in-memory implementation. The stored preferences are available while the application process is running, but they are not persisted after restarting the application.

---

## 👤 Human-in-the-Loop

The project uses LangGraph's:

```python
interrupt()
```

When the assistant detects a learning preference, it does not save it immediately.

Instead, it calls:

```text
propose_preference
```

The graph then pauses and asks the user for approval.

### Example

```text
User:
I prefer practical examples.

        ↓

AI detects a learning preference

        ↓

propose_preference

        ↓

Human Review

   ┌───────────────┐
   │ Approve       │
   │      or       │
   │ Reject        │
   └───────┬───────┘
           │
     ┌─────┴─────┐
     ▼           ▼
   Save       Don't Save
```

The conversation resumes using:

```python
Command(resume={...})
```

This demonstrates a real **Human-in-the-Loop workflow** where the human controls what information is stored as long-term memory.

---

## 🔧 Tools

The assistant has three main tools.

### `search_web`

Searches the web for:

- AI resources
- Machine Learning tutorials
- Programming documentation
- Technical articles
- Learning resources

### `scrape_page`

Fetches and extracts readable content from a web page.

### `propose_preference`

Detects a possible learning preference.

This tool does **not** save the preference itself.

Instead, the graph sends the request to the Human-in-the-Loop review step.

---

## 📁 Project Structure

```text
AI_Study_Assistant/
│
├── app.py
├── graph.py
├── memory.py
├── state.py
├── tools.py
├── config.py
├── util.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies

- Python
- LangGraph
- LangChain
- Google Gemini
- Streamlit
- BeautifulSoup
- Requests
- Pydantic

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_Study_Assistant_LangGraph.git
```

```bash
cd AI_Study_Assistant_LangGraph
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy:

```text
.env.example
```

to:

```text
.env
```

Then add your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Never commit `.env` to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Testing the Memory System

### Test Short-Term Memory

Ask:

```text
What is RAG?
```

Then:

```text
What are the two main components you mentioned?
```

The assistant should use the context from the current conversation.

---

### Test Human-in-the-Loop

Tell the assistant:

```text
I prefer practical examples when learning technical topics.
```

The assistant should ask for approval before saving the preference.

Select:

```text
✅ Approve & remember
```

---

### Test Long-Term Memory

Start:

```text
🔄 New conversation (same user)
```

Then ask:

```text
Explain Machine Learning to me.
```

The assistant should still be aware of the saved learning preference even though the conversation has a new `thread_id`.

---

### Test Rejection

Tell the assistant:

```text
I prefer concise explanations.
```

Select:

```text
❌ Reject
```

The preference should not be added to long-term memory.

---

## 🔄 Memory Comparison

| Feature | Short-Term Memory | Long-Term Memory |
|---|---|---|
| Implementation | `MemorySaver` | `InMemoryStore` |
| Scope | Conversation / Thread | User |
| Identifier | `thread_id` | `user_id` |
| New conversation | Resets context | Preferences remain |
| Example | Previous messages | Learning preferences |

---

## 🎓 Main Concepts Demonstrated

This project demonstrates three important LangGraph concepts:

### Short-Term Memory

Maintaining conversation state using a checkpointer and `thread_id`.

### Long-Term Memory

Storing reusable user information using a store and `user_id`.

### Human-in-the-Loop

Pausing graph execution with `interrupt()` and allowing the user to approve or reject an action before continuing.

---

## ⚠️ Limitations

The current project uses:

```python
InMemoryStore()
```

for long-term memory.

Therefore, memories are not persistent after the application process stops.

For a production system, the memory layer could be replaced with a persistent database or storage solution.

---

## 👩‍💻 Author

**Reem Mohamed**

AI / Computer Vision / Machine Learning Developer

Interests:

- Artificial Intelligence
- Computer Vision
- Machine Learning
- Deep Learning
- LLMs
- AI Agents
- RAG
- PyTorch