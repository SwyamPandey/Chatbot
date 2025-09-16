import streamlit as st
from rr import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def generate_thread_name(first_message):
    """Generate a friendly thread name from the first message"""
    if not first_message:
        return "New Chat"
    
    # Truncate and clean the message for display
    name = first_message.strip()
    if len(name) > 50:
        name = name[:47] + "..."
    
    # Remove newlines and extra spaces
    name = " ".join(name.split())
    
    return name if name else "New Chat"

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id, "New Chat")  # Start with placeholder name
    st.session_state["message_history"] = []

def add_thread(thread_id, thread_name="New Chat"):
    if thread_id not in [t["id"] for t in st.session_state["chat_threads"]]:
        st.session_state["chat_threads"].append({
            "id": thread_id,
            "name": thread_name
        })

def update_thread_name(thread_id, new_name):
    """Update the name of an existing thread"""
    for thread in st.session_state["chat_threads"]:
        if thread["id"] == thread_id:
            thread["name"] = new_name
            break

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

def get_thread_name_from_existing_threads():
    """Convert existing UUID-only threads to the new format"""
    converted_threads = []
    existing_threads = retrieve_all_threads()
    
    for thread_id in existing_threads:
        # Try to get the first message to generate a name
        messages = load_conversation(thread_id)
        first_message = ""
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                first_message = msg.content
                break
        
        thread_name = generate_thread_name(first_message)
        converted_threads.append({
            "id": thread_id,
            "name": thread_name
        })
    
    return converted_threads

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Convert existing threads or start fresh
    st.session_state["chat_threads"] = get_thread_name_from_existing_threads()

# Add current thread if not exists
current_thread_exists = any(t["id"] == st.session_state["thread_id"] for t in st.session_state["chat_threads"])
if not current_thread_exists:
    add_thread(st.session_state["thread_id"], "New Chat")

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread in st.session_state["chat_threads"][::-1]:  # Show newest first
    thread_id = thread["id"]
    thread_name = thread["name"]
    
    if st.sidebar.button(thread_name, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Update thread name if this is the first message in a new chat
    current_thread = next((t for t in st.session_state["chat_threads"] if t["id"] == st.session_state["thread_id"]), None)
    if current_thread and current_thread["name"] == "New Chat" and not st.session_state["message_history"]:
        new_name = generate_thread_name(user_input)
        update_thread_name(st.session_state["thread_id"], new_name)
    
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
