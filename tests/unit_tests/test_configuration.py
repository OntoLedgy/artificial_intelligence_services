from langgraph.pregel import Pregel

from agents.langgraph.langgraph_server.src.agent.graph import graph


def test_placeholder() -> None:
    # TODO: You can add actual unit tests
    # for your graph and other logic here.
    assert isinstance(graph, Pregel)
