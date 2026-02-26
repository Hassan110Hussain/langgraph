from numpy import isin
import streamlit as st

# from langgraph_backend import chatbot
from langgraph_backend_tool import chatbot
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import uuid

DEFAULT_TOPIC = "New chat"
MAX_TOPIC_LEN = 60


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def truncate_topic(topic: str) -> str:
    topic = topic.strip()
    if len(topic) > MAX_TOPIC_LEN:
        return topic[: MAX_TOPIC_LEN - 3] + "..."
    return topic


def derive_topic_from_messages(messages):
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content.strip()
            if content:
                return truncate_topic(content)
    return None


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    st.session_state["thread_topics"][thread_id] = DEFAULT_TOPIC


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_convo(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    values = state.values

    return values.get("messages", [])


def extract_tool_name(message_chunk, metadata):
    if isinstance(message_chunk, ToolMessage):
        return message_chunk.name

    tool_calls = getattr(message_chunk, "tool_calls", None)
    if tool_calls:
        first_call = tool_calls[0] or {}
        if "name" in first_call:
            return first_call["name"]
        function_call = first_call.get("function")
        if function_call:
            return function_call.get("name")

    return metadata.get("tool_name") or metadata.get("langgraph_node")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "thread_topics" not in st.session_state:
    st.session_state["thread_topics"] = {}

add_thread(st.session_state["thread_id"])
st.session_state["thread_topics"].setdefault(
    st.session_state["thread_id"], DEFAULT_TOPIC
)

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My convos")

for thread_id in st.session_state["chat_threads"][::-1]:
    thread_title = st.session_state["thread_topics"].get(thread_id, str(thread_id))
    if st.sidebar.button(thread_title):
        st.session_state["thread_id"] = thread_id
        messages = load_convo(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages
        topic = derive_topic_from_messages(messages)
        if topic:
            st.session_state["thread_topics"][thread_id] = topic

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("type here")

if user_input:

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    if (
        st.session_state["thread_topics"].get(st.session_state["thread_id"])
        in (None, DEFAULT_TOPIC)
        and user_input.strip()
    ):
        st.session_state["thread_topics"][st.session_state["thread_id"]] = (
            truncate_topic(user_input)
        )

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    assistant_container = st.chat_message("assistant")
    status_container = assistant_container.status("Thinking...", state="running")

    def ai_only_stream():
        current_tool = None
        try:
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                tool_name = extract_tool_name(message_chunk, metadata or {})
                if tool_name and tool_name != current_tool:
                    current_tool = tool_name
                    status_container.update(
                        label=f"Using tool: {tool_name}", state="running"
                    )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        finally:
            status_container.update(label="Answer ready", state="complete")

    ai_message = assistant_container.write_stream(ai_only_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
