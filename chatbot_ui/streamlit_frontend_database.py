import streamlit as st
from langgraph_backend_database import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def add_thread(thread_id):
    threads = st.session_state["chat_threads"]
    if thread_id in threads:
        threads.remove(thread_id)
    threads.insert(0, thread_id)


def load_convo(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    values = state.values

    return values.get("messages", [])


def get_thread_title(thread_id, messages):
    for msg in messages:
        if isinstance(msg, HumanMessage):
            title = msg.content.strip()
            if not title:
                continue
            title = title.splitlines()[0]
            if len(title) > 40:
                title = f"{title[:37]}..."
            return title

    return "New chat"


def build_session_messages(messages):
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        formatted_messages.append({"role": role, "content": msg.content})
    return formatted_messages


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My convos")

for thread_id in st.session_state["chat_threads"]:
    messages = load_convo(thread_id)
    title = get_thread_title(thread_id, messages)
    if st.sidebar.button(title, key=f"thread-{thread_id}"):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = build_session_messages(messages)

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("type here")

if user_input:

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state['thread_id']
        },
        "run_name": "chat_turn",
        }

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
