from __future__ import annotations

"""Neo4j-backed bibliographic graph with a graceful disabled mode.

The graph stores only paper metadata. If Neo4j is not configured or reachable,
callers receive an empty graph result and the normal RAG path remains usable.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config


REL_WRITTEN_BY = "WRITTEN_BY"
REL_PUBLISHED_IN = "PUBLISHED_IN"
REL_PUBLISHED_YEAR = "PUBLISHED_YEAR"
REL_IN_FIELD = "IN_FIELD"
REL_HAS_KEYWORD = "HAS_KEYWORD"
REL_WORKS_ON = "WORKS_ON"

_FIELD_ALIASES = {
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision",
    "cv": "Computer Vision",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "reinforcement learning": "Reinforcement Learning",
    "information retrieval": "Information Retrieval",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "rag": "Retrieval-Augmented Generation",
    "large language models": "Large Language Models",
    "large language model": "Large Language Models",
    "llm": "Large Language Models",
    "graph neural networks": "Graph Neural Networks",
    "graph neural network": "Graph Neural Networks",
    "gnn": "Graph Neural Networks",
    "knowledge graph": "Knowledge Graphs",
    "knowledge graphs": "Knowledge Graphs",
    "recommendation": "Recommender Systems",
    "recommender systems": "Recommender Systems",
    "multimodal learning": "Multimodal Learning",
    "speech": "Speech Processing",
    "robotics": "Robotics",
    "data mining": "Data Mining",
    "security": "Computer Security",
    "database": "Databases",
    "databases": "Databases",
    "distributed systems": "Distributed Systems",
    "software engineering": "Software Engineering",
}

_FIELD_HINTS = {
    "Natural Language Processing": ("natural language", "nlp", "text generation", "translation", "named entity"),
    "Computer Vision": ("computer vision", "image", "visual", "object detection", "segmentation"),
    "Machine Learning": ("machine learning", "classification", "regression", "generalization"),
    "Deep Learning": ("deep learning", "neural network", "transformer", "representation learning"),
    "Reinforcement Learning": ("reinforcement learning", "policy learning", "reward model"),
    "Information Retrieval": ("information retrieval", "document retrieval", "search engine", "ranking"),
    "Retrieval-Augmented Generation": ("retrieval-augmented", "retrieval augmented", "graphrag"),
    "Large Language Models": ("large language model", "llm", "language model"),
    "Graph Neural Networks": ("graph neural", "message passing network", "gnn"),
    "Knowledge Graphs": ("knowledge graph", "graph reasoning", "entity relation"),
    "Recommender Systems": ("recommender", "recommendation system", "collaborative filtering"),
    "Multimodal Learning": ("multimodal", "vision-language", "image-text"),
    "Speech Processing": ("speech recognition", "speech synthesis", "audio"),
    "Robotics": ("robot", "robotics", "manipulation", "navigation"),
    "Data Mining": ("data mining", "pattern mining", "anomaly detection"),
}

_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "paper", "the", "this", "to", "what", "when",
    "which", "with", "year", "论文", "这篇", "这个", "作者", "哪些", "还有", "什么",
    "相关", "工作", "研究", "领域", "发表", "哪年", "年份",
}


class GraphUnavailableError(RuntimeError):
    """Raised when the optional Neo4j graph backend cannot be used."""


def _normalise(value: str | None) -> str:
    text = (value or "").casefold().replace("&", " and ")
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_list(values) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = re.split(r"[,;|]", values)
    result = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalise_field(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    return _FIELD_ALIASES.get(_normalise(cleaned), cleaned)


def _infer_fields(meta: dict) -> list[str]:
    explicit = [_normalise_field(value) for value in _clean_list(meta.get("fields"))]
    if explicit:
        return list(dict.fromkeys(explicit))[:4]
    haystack = f" {_normalise((meta.get('title') or '') + ' ' + (meta.get('abstract') or ''))} "
    return [
        field for field, hints in _FIELD_HINTS.items()
        if any(_normalise(hint) in haystack for hint in hints)
    ][:4]


def _coerce_year(value, meta: dict) -> int | None:
    if isinstance(value, int) and 1800 <= value <= 2200:
        return value
    match = re.search(r"\b(18|19|20|21)\d{2}\b", str(value or ""))
    if not match:
        source_text = " ".join(str(meta.get(key) or "") for key in ("journal", "source", "pdf_stem"))
        match = re.search(r"\b(18|19|20|21)\d{2}\b", source_text)
    return int(match.group(0)) if match else None


def prepare_graph_metadata(meta: dict) -> dict:
    """Normalise metadata before it is written to Neo4j."""
    result = dict(meta)
    result["source"] = str(meta.get("source") or "").strip()
    result["title"] = re.sub(r"\s+", " ", str(meta.get("title") or "")).strip() or None
    result["authors"] = _clean_list(meta.get("authors"))
    result["journal"] = re.sub(r"\s+", " ", str(meta.get("journal") or "")).strip() or None
    result["year"] = _coerce_year(meta.get("year"), meta)
    result["fields"] = _infer_fields(meta)
    result["keywords"] = _clean_list(meta.get("keywords"))[:12]
    return result


@dataclass
class _Paper:
    source: str
    title: str
    year: int | None
    venue: str | None
    abstract: str
    authors: list[str]
    fields: list[str]
    keywords: list[str]


class LiteratureGraph:
    """Neo4j implementation of the literature graph repository."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        namespace: str | None = None,
        driver=None,
    ):
        self.uri = uri or config.NEO4J_URI
        self.username = username or config.NEO4J_USERNAME
        self.password = password or config.NEO4J_PASSWORD
        self.database = database or config.NEO4J_DATABASE
        self.namespace = namespace or config.NEO4J_NAMESPACE
        self._driver = driver

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if not self.uri or not self.password:
            raise GraphUnavailableError("Neo4j URI or password is not configured")
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:
            raise GraphUnavailableError("Neo4j Python driver is not installed") from exc
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self._driver.verify_connectivity()
            return self._driver
        except Exception as exc:
            self._driver = None
            raise GraphUnavailableError(f"Neo4j is unavailable: {exc}") from exc

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _read(self, query: str, **params) -> list[dict]:
        driver = self._get_driver()
        try:
            with driver.session(database=self.database) as session:
                def run(tx):
                    return [record.data() for record in tx.run(query, namespace=self.namespace, **params)]

                if hasattr(session, "execute_read"):
                    return session.execute_read(run)
                return run(session)
        except GraphUnavailableError:
            raise
        except Exception as exc:
            raise GraphUnavailableError(f"Neo4j read failed: {exc}") from exc

    def _write(self, callback) -> Any:
        driver = self._get_driver()
        try:
            with driver.session(database=self.database) as session:
                if hasattr(session, "execute_write"):
                    return session.execute_write(callback)
                return callback(session)
        except GraphUnavailableError:
            raise
        except Exception as exc:
            raise GraphUnavailableError(f"Neo4j write failed: {exc}") from exc

    def initialize(self) -> None:
        statements = [
            "CREATE INDEX re0rag_paper_lookup IF NOT EXISTS FOR (p:Paper) ON (p.namespace, p.source)",
            "CREATE INDEX re0rag_author_lookup IF NOT EXISTS FOR (a:Author) ON (a.namespace, a.canonical_name)",
            "CREATE INDEX re0rag_field_lookup IF NOT EXISTS FOR (f:Field) ON (f.namespace, f.canonical_name)",
        ]

        def create(tx):
            for statement in statements:
                tx.run(statement).consume()

        self._write(create)

    def upsert_paper(self, meta: dict) -> str:
        cleaned = prepare_graph_metadata(meta)
        source = cleaned["source"]
        if not source:
            raise ValueError("Paper metadata must contain source")
        self.initialize()
        title = cleaned.get("title") or Path(source).stem
        props = {
            "source": source,
            "title": title,
            "year": cleaned.get("year"),
            "venue": cleaned.get("journal"),
            "abstract": cleaned.get("abstract") or "",
            "authors": cleaned.get("authors") or [],
            "fields": cleaned.get("fields") or [],
            "keywords": cleaned.get("keywords") or [],
        }

        def write(tx):
            tx.run(
                """
                MERGE (p:Paper:Re0Rag {namespace: $namespace, source: $source})
                SET p.title = $title, p.year = $year, p.venue = $venue,
                    p.abstract = $abstract, p.authors = $authors, p.fields = $fields,
                    p.keywords = $keywords
                WITH p
                OPTIONAL MATCH (p)-[old]->()
                WHERE type(old) IN $relations
                DELETE old
                """,
                namespace=self.namespace,
                **props,
                relations=[REL_WRITTEN_BY, REL_PUBLISHED_IN, REL_PUBLISHED_YEAR, REL_IN_FIELD, REL_HAS_KEYWORD],
            ).consume()
            if props["authors"]:
                tx.run(
                    """
                    MATCH (p:Paper:Re0Rag {namespace: $namespace, source: $source})
                    UNWIND range(0, size($authors) - 1) AS position
                    WITH p, position, $authors[position] AS author_name
                    MERGE (a:Author:Re0Rag {namespace: $namespace, canonical_name: toLower(author_name)})
                    SET a.name = author_name
                    MERGE (p)-[r:WRITTEN_BY]->(a)
                    SET r.position = position, r.namespace = $namespace
                    """,
                    namespace=self.namespace,
                    source=source,
                    authors=props["authors"],
                ).consume()
            if props["venue"]:
                tx.run(
                    """
                    MATCH (p:Paper:Re0Rag {namespace: $namespace, source: $source})
                    MERGE (v:Venue:Re0Rag {namespace: $namespace, canonical_name: toLower($venue)})
                    SET v.name = $venue
                    MERGE (p)-[r:PUBLISHED_IN]->(v)
                    SET r.namespace = $namespace
                    """,
                    namespace=self.namespace,
                    source=source,
                    venue=props["venue"],
                ).consume()
            if props["year"]:
                tx.run(
                    """
                    MATCH (p:Paper:Re0Rag {namespace: $namespace, source: $source})
                    MERGE (y:Year:Re0Rag {namespace: $namespace, value: $year})
                    SET y.name = toString($year)
                    MERGE (p)-[r:PUBLISHED_YEAR]->(y)
                    SET r.namespace = $namespace
                    """,
                    namespace=self.namespace,
                    source=source,
                    year=props["year"],
                ).consume()
            self._write_named_entities(tx, source, props["fields"], "Field", REL_IN_FIELD)
            self._write_named_entities(tx, source, props["keywords"], "Keyword", REL_HAS_KEYWORD)
            self._rebuild_author_field_edges(tx)
            self._remove_orphans(tx)

        self._write(write)
        return source

    def _write_named_entities(self, tx, source: str, values: list[str], label: str, relation: str) -> None:
        if not values:
            return
        tx.run(
            f"""
            MATCH (p:Paper:Re0Rag {{namespace: $namespace, source: $source}})
            UNWIND $values AS value
            MERGE (entity:{label}:Re0Rag {{namespace: $namespace, canonical_name: toLower(value)}})
            SET entity.name = value
            MERGE (p)-[r:{relation}]->(entity)
            SET r.namespace = $namespace
            """,
            namespace=self.namespace,
            source=source,
            values=values,
        ).consume()

    def delete_paper(self, source: str) -> bool:
        self.initialize()

        def write(tx):
            result = tx.run(
                """
                MATCH (p:Paper:Re0Rag {namespace: $namespace, source: $source})
                RETURN count(p) AS count
                """,
                namespace=self.namespace,
                source=source,
            ).single()
            tx.run(
                "MATCH (p:Paper:Re0Rag {namespace: $namespace, source: $source}) DETACH DELETE p",
                namespace=self.namespace,
                source=source,
            ).consume()
            self._rebuild_author_field_edges(tx)
            self._remove_orphans(tx)
            return bool(result and result["count"])

        return self._write(write)

    def rebuild(self, metadata_dir: str | Path) -> dict:
        metadata_dir = Path(metadata_dir)
        self.initialize()

        def clear(tx):
            tx.run(
                "MATCH (n:Re0Rag {namespace: $namespace}) DETACH DELETE n",
                namespace=self.namespace,
            ).consume()

        self._write(clear)
        indexed = 0
        skipped = 0
        for path in sorted(metadata_dir.glob("*_meta.json")):
            try:
                self.upsert_paper(json.loads(path.read_text(encoding="utf-8")))
                indexed += 1
            except Exception:
                skipped += 1
        return {"indexed": indexed, "skipped": skipped, **self.stats()}

    def stats(self) -> dict:
        node_rows = self._read(
            "MATCH (n:Re0Rag {namespace: $namespace}) UNWIND labels(n) AS label "
            "WITH label WHERE label <> 'Re0Rag' RETURN label AS node_type, count(*) AS count"
        )
        relation_rows = self._read(
            "MATCH (:Re0Rag {namespace: $namespace})-[r]->(:Re0Rag {namespace: $namespace}) "
            "RETURN type(r) AS relation, count(*) AS count"
        )
        return {
            "nodes": {row["node_type"]: row["count"] for row in node_rows},
            "edges": sum(row["count"] for row in relation_rows),
            "relations": {row["relation"]: row["count"] for row in relation_rows},
        }

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        papers = self._load_papers()
        if not papers:
            return []
        intent = _detect_intent(query)
        ranked = sorted(((self._seed_score(query, paper), paper) for paper in papers), reverse=True, key=lambda item: item[0])
        seed_score, seed = ranked[0]
        seed = seed if seed_score >= 2.5 else None
        if seed and intent == "year":
            path = f"Paper({seed.title}) -> Year({seed.year})" if seed.year else f"Paper({seed.title}) has no indexed year"
            return [self._evidence(seed, seed_score, [path])]

        candidate_sources = self._traverse_sources(seed, intent) if seed else None
        candidates = []
        for paper in papers:
            if candidate_sources is not None and paper.source not in candidate_sources:
                continue
            if seed and paper.source == seed.source and intent in {"author_works", "field_related"}:
                continue
            score, paths = self._related_score(query, paper, seed, intent)
            if score > 0:
                candidates.append((score, paper, paths))
        candidates.sort(key=lambda item: (-item[0], item[1].year or 0, item[1].title))
        result = [self._evidence(paper, score, paths) for score, paper, paths in candidates[:top_k]]
        if seed and intent == "metadata" and not result:
            result = [self._evidence(seed, seed_score, [f"Matched Paper({seed.title})"])]
        return result

    def _load_papers(self) -> list[_Paper]:
        rows = self._read(
            """
            MATCH (p:Paper:Re0Rag {namespace: $namespace})
            RETURN p.source AS source, p.title AS title, p.year AS year, p.venue AS venue,
                   p.abstract AS abstract, p.authors AS authors, p.fields AS fields, p.keywords AS keywords
            """
        )
        return [
            _Paper(
                source=row.get("source") or "",
                title=row.get("title") or row.get("source") or "Unknown paper",
                year=row.get("year"), venue=row.get("venue"), abstract=row.get("abstract") or "",
                authors=row.get("authors") or [], fields=row.get("fields") or [], keywords=row.get("keywords") or [],
            )
            for row in rows
        ]

    def _traverse_sources(self, seed: _Paper, intent: str) -> set[str] | None:
        if intent == "author_works":
            query = (
                "MATCH (seed:Paper:Re0Rag {namespace: $namespace, source: $source})"
                "-[:WRITTEN_BY]->(:Author:Re0Rag)<-[:WRITTEN_BY]-(candidate:Paper:Re0Rag {namespace: $namespace}) "
                "WHERE candidate <> seed RETURN DISTINCT candidate.source AS source"
            )
        elif intent == "field_related":
            query = (
                "MATCH (seed:Paper:Re0Rag {namespace: $namespace, source: $source})"
                "-[:IN_FIELD]->(:Field:Re0Rag)<-[:IN_FIELD]-(candidate:Paper:Re0Rag {namespace: $namespace}) "
                "WHERE candidate <> seed RETURN DISTINCT candidate.source AS source"
            )
        else:
            return None
        return {row["source"] for row in self._read(query, source=seed.source)}

    @staticmethod
    def _seed_score(query: str, paper: _Paper) -> float:
        query_norm = _normalise(query)
        title_norm = _normalise(paper.title)
        source_norm = _normalise(Path(paper.source).stem)
        if title_norm and title_norm in query_norm:
            return 100.0
        if source_norm and source_norm in query_norm:
            return 80.0
        query_tokens, title_tokens = set(_query_tokens(query)), set(_query_tokens(paper.title))
        overlap = query_tokens & title_tokens
        score = 4.0 * len(overlap) + 8.0 * len(overlap) / max(1, len(title_tokens))
        for author in paper.authors:
            if _normalise(author) in query_norm:
                score += 5.0
            elif query_tokens & set(_query_tokens(author)):
                score += 2.0
        return score + 3.0 * len(set(_query_fields(query)) & set(paper.fields))

    @classmethod
    def _related_score(cls, query: str, paper: _Paper, seed: _Paper | None, intent: str) -> tuple[float, list[str]]:
        direct = cls._seed_score(query, paper)
        if seed is None:
            return direct, [f"Direct metadata match -> Paper({paper.title})"] if direct > 0 else []
        shared_authors = sorted(set(seed.authors) & set(paper.authors))
        shared_fields = sorted(set(seed.fields) & set(paper.fields))
        shared_keywords = sorted(set(seed.keywords) & set(paper.keywords))
        same_venue = bool(seed.venue and paper.venue and _normalise(seed.venue) == _normalise(paper.venue))
        if intent == "author_works" and not shared_authors:
            return 0.0, []
        if intent == "field_related" and not shared_fields:
            return 0.0, []
        score, paths = 0.0, []
        for author in shared_authors:
            score += 8.0 if intent == "author_works" else 5.0
            paths.append(f"Paper({seed.title}) -> Author({author}) -> Paper({paper.title})")
            paths.extend(f"Paper({seed.title}) -> Author({author}) -> Field({field}) -> Paper({paper.title})" for field in shared_fields)
        for field in shared_fields:
            score += 6.0 if intent == "field_related" else 2.5
            paths.append(f"Paper({seed.title}) -> Field({field}) -> Paper({paper.title})")
        if same_venue:
            score += 0.75
            paths.append(f"Paper({seed.title}) -> Venue({seed.venue}) -> Paper({paper.title})")
        if shared_keywords:
            score += min(2.0, 0.5 * len(shared_keywords))
        if intent == "metadata" and paper.source == seed.source:
            score += 50.0
            paths.append(f"Matched Paper({paper.title})")
        return score, paths

    @staticmethod
    def _evidence(paper: _Paper, score: float, paths: list[str]) -> dict:
        facts = [f"Title: {paper.title}"]
        if paper.authors:
            facts.append("Authors: " + ", ".join(paper.authors))
        if paper.year:
            facts.append(f"Year: {paper.year}")
        if paper.venue:
            facts.append(f"Venue: {paper.venue}")
        if paper.fields:
            facts.append("Fields: " + ", ".join(paper.fields))
        if paper.keywords:
            facts.append("Keywords: " + ", ".join(paper.keywords))
        if paper.abstract:
            facts.append("Abstract: " + paper.abstract)
        if paths:
            facts.append("Graph paths: " + " | ".join(paths))
        return {
            "doc_type": "graph", "content": "\n".join(facts), "graph_paths": paths,
            "score": round(float(score), 4),
            "metadata": {"source": paper.source, "title": paper.title, "authors": paper.authors,
                         "year": paper.year, "journal": paper.venue, "fields": paper.fields,
                         "keywords": paper.keywords},
        }

    def _rebuild_author_field_edges(self, tx) -> None:
        tx.run(
            "MATCH (:Re0Rag {namespace: $namespace})-[r:WORKS_ON]->() DELETE r",
            namespace=self.namespace,
        ).consume()
        tx.run(
            """
            MATCH (p:Paper:Re0Rag {namespace: $namespace})-[:WRITTEN_BY]->(a:Author:Re0Rag {namespace: $namespace}),
                  (p)-[:IN_FIELD]->(f:Field:Re0Rag {namespace: $namespace})
            WITH a, f, count(DISTINCT p) AS weight
            MERGE (a)-[r:WORKS_ON]->(f)
            SET r.weight = weight, r.namespace = $namespace
            """
            ,
            namespace=self.namespace,
        ).consume()

    def _remove_orphans(self, tx) -> None:
        tx.run(
            """
            MATCH (n:Re0Rag {namespace: $namespace})
            WHERE NOT n:Paper AND NOT (n)--()
            DELETE n
            """
            ,
            namespace=self.namespace,
        ).consume()


def _query_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", _normalise(text))
    return [token for token in tokens if token not in _QUERY_STOP_WORDS and len(token) > 1]


def _query_fields(text: str) -> list[str]:
    query = f" {_normalise(text)} "
    matched = []
    for alias, canonical in _FIELD_ALIASES.items():
        alias_norm = _normalise(alias)
        if re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", query):
            matched.append(canonical)
    return list(dict.fromkeys(matched))


def _detect_intent(query: str) -> str:
    q = _normalise(query)
    if re.search(r"哪一?年|年份|何时|when|year|published in", q):
        return "year"
    if re.search(r"作者.*(?:论文|工作|研究)|相关工作|other (?:papers|work)|author.*(?:papers|work|publication)", q):
        return "author_works"
    if re.search(r"领域|方向|同类研究|其他研究|related research|same field|research area", q):
        return "field_related"
    return "metadata"


_default_graph: LiteratureGraph | None = None
_backend_status: dict[str, Any] | None = None


def _resolve_graph() -> LiteratureGraph | None:
    global _default_graph, _backend_status
    if _backend_status is not None:
        return _default_graph
    if not config.GRAPH_ENABLED:
        _backend_status = {"enabled": False, "backend": "disabled", "reason": "RE0RAG_GRAPH_ENABLED=0"}
        return None
    if config.GRAPH_BACKEND != "neo4j":
        _backend_status = {"enabled": False, "backend": config.GRAPH_BACKEND, "reason": "Only neo4j is supported"}
        return None
    graph = LiteratureGraph()
    try:
        graph.initialize()
    except GraphUnavailableError as exc:
        _backend_status = {"enabled": False, "backend": "neo4j", "reason": str(exc)}
        return None
    _default_graph = graph
    _backend_status = {"enabled": True, "backend": "neo4j", "reason": ""}
    return graph


def graph_status() -> dict[str, Any]:
    _resolve_graph()
    return dict(_backend_status or {"enabled": False, "backend": "disabled", "reason": "Unavailable"})


def is_graph_enabled() -> bool:
    return bool(graph_status()["enabled"])


def _mark_unavailable(error: GraphUnavailableError) -> None:
    global _default_graph, _backend_status
    if _default_graph is not None:
        try:
            _default_graph.close()
        except Exception:
            pass
    _default_graph = None
    _backend_status = {"enabled": False, "backend": "neo4j", "reason": str(error)}


def upsert_paper(meta: dict) -> str | None:
    graph = _resolve_graph()
    if not graph:
        return None
    try:
        return graph.upsert_paper(meta)
    except GraphUnavailableError as exc:
        _mark_unavailable(exc)
        return None


def delete_paper(source: str) -> bool:
    graph = _resolve_graph()
    if not graph:
        return False
    try:
        return graph.delete_paper(source)
    except GraphUnavailableError as exc:
        _mark_unavailable(exc)
        return False


def rebuild_graph(metadata_dir: str | Path | None = None) -> dict:
    graph = _resolve_graph()
    if not graph:
        return {"indexed": 0, "skipped": 0, "nodes": {}, "edges": 0, "relations": {}, **graph_status()}
    try:
        return {**graph.rebuild(metadata_dir or config.META_DIR), **graph_status()}
    except GraphUnavailableError as exc:
        _mark_unavailable(exc)
        return {"indexed": 0, "skipped": 0, "nodes": {}, "edges": 0, "relations": {}, **graph_status()}


def graph_stats() -> dict:
    graph = _resolve_graph()
    if not graph:
        return {"nodes": {}, "edges": 0, "relations": {}, **graph_status()}
    try:
        return {**graph.stats(), **graph_status()}
    except GraphUnavailableError as exc:
        _mark_unavailable(exc)
        return {"nodes": {}, "edges": 0, "relations": {}, **graph_status()}


def search_graph(query: str, top_k: int | None = None) -> list[dict]:
    graph = _resolve_graph()
    if not graph:
        return []
    try:
        return graph.search(query, top_k=top_k or config.GRAPH_TOP_K)
    except GraphUnavailableError as exc:
        _mark_unavailable(exc)
        return []
