from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[
        list,
        add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model(
        "openai:gpt-4.1")


def chatbot(state: State):
    messages_dictionary = {
        "messages": [
            llm.invoke(
                    state["messages"]
                    )
            ]
        }
    
    return messages_dictionary


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node(
        "chatbot",
        chatbot)

graph_builder.add_edge(
        START,
        "chatbot")

graph = graph_builder.compile()