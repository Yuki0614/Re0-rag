import db.literature_graph as graph_module


def _meta():
    return {
        "source": "alpha.md",
        "title": "Alpha Retrieval",
        "authors": ["Alice Smith", "Bob Lee"],
        "year": 2023,
        "journal": "Test Conference",
        "fields": ["Information Retrieval", "RAG"],
        "keywords": ["retrieval"],
        "abstract": "A retrieval-augmented generation method.",
    }


class _Result:
    def consume(self):
        return self


class _Tx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return _Result()


class _Session:
    def __init__(self, tx):
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute_write(self, callback):
        return callback(self.tx)


class _Driver:
    def __init__(self):
        self.tx = _Tx()

    def session(self, **_kwargs):
        return _Session(self.tx)


def test_neo4j_upsert_creates_metadata_nodes_and_edges():
    driver = _Driver()
    graph = graph_module.LiteratureGraph(driver=driver, namespace="test")

    assert graph.upsert_paper(_meta()) == "alpha.md"

    statements = "\n".join(query for query, _params in driver.tx.calls)
    assert "MERGE (p:Paper:Re0Rag" in statements
    assert "MERGE (a:Author:Re0Rag" in statements
    assert "MERGE (entity:Field:Re0Rag" in statements
    assert "MERGE (entity:Keyword:Re0Rag" in statements
    assert "WORKS_ON" in statements
    assert all(params.get("namespace") == "test" for _query, params in driver.tx.calls if params)


def test_disabled_graph_never_attempts_neo4j(monkeypatch):
    monkeypatch.setattr(graph_module.config, "GRAPH_ENABLED", False)
    monkeypatch.setattr(graph_module, "_default_graph", None)
    monkeypatch.setattr(graph_module, "_backend_status", None)

    status = graph_module.graph_status()
    assert status["enabled"] is False
    assert graph_module.search_graph("Alpha Retrieval 是哪一年？") == []
    assert graph_module.upsert_paper(_meta()) is None
