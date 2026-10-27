from app import rag


def test_retrieves_right_topic():
    assert rag.retrieve("how can I sleep better")[0]["topic"] == "sleep"
    assert rag.retrieve("akelapan kaise dur karun")[0]["topic"] == "loneliness"


def test_out_of_scope_returns_nothing():
    assert rag.retrieve("capital of France") == []
    assert rag.retrieve("write python code to sort a list") == []


def test_context_block_empty_when_nothing_found():
    assert rag.context_block("bitcoin price today") == ""
