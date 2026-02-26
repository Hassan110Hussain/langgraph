from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT thread_id, MAX(rowid) AS last_rowid
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY last_rowid DESC
        """
    )
    rows = cursor.fetchall()
    return [row[0] for row in rows]
