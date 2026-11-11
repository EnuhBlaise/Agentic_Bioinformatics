"""Reflection memory (Mnemo-style): lessons written after runs, retrieved before runs.

Backends: `local` (JSONL + embeddings in numpy; no network, no model
download) and `chroma` (ChromaDB persistent client). Embeddings: `hashing`
(deterministic feature hashing of word uni/bi-grams; works air-gapped) or
`ollama` (local embedding model).

Lessons have a lifecycle (candidate -> validated | rejected). An agent that
writes its own rules can entrench its own mistakes, so a candidate is only
promoted by a human (API/CLI) or by being re-derived in N independent runs.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np

from .config import MemoryConfig
from .state import Lesson

DIM = 512
AUTO_VALIDATE_AFTER = 3  # independent re-derivations
DEDUP_SIM = 0.85


def _tokens(text: str) -> list[str]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


def hashing_embed(text: str) -> np.ndarray:
    v = np.zeros(DIM)
    for t in _tokens(text):
        h = int.from_bytes(hashlib.md5(t.encode()).digest()[:8], "little")
        v[h % DIM] += 1.0 if (h >> 63) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class Embedder:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg

    def __call__(self, text: str) -> np.ndarray:
        if self.cfg.embedding == "ollama":
            import httpx

            r = httpx.post(self.cfg.embedding_base_url.rstrip("/") + "/api/embeddings",
                           json={"model": self.cfg.embedding_model, "prompt": text}, timeout=60)
            r.raise_for_status()
            v = np.asarray(r.json()["embedding"], float)
            return v / (np.linalg.norm(v) or 1)
        return hashing_embed(text)


class MemoryStore:
    def add(self, lesson: Lesson) -> Lesson: ...
    def search(self, query: str, k: int, include_candidates: bool = True) -> list[Lesson]: ...
    def all(self) -> list[Lesson]: ...
    def set_status(self, lesson_id: str, status: str) -> Lesson: ...


class LocalMemory(MemoryStore):
    def __init__(self, cfg: MemoryConfig, root: Path):
        self.cfg = cfg
        self.path = root / "lessons.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embed = Embedder(cfg)
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def _write(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(self.path)

    def add(self, lesson: Lesson) -> Lesson:
        with self._lock:
            rows = self._load()
            v = self.embed(lesson.text)
            for r in rows:
                if r["lesson"]["status"] == "rejected":
                    continue
                if float(np.dot(v, np.asarray(r["vec"]))) >= DEDUP_SIM:
                    # Re-derived independently: count support instead of duplicating.
                    srcs = set(r.get("support_runs", [])) | {lesson.source_run}
                    r["support_runs"] = sorted(s for s in srcs if s)
                    if r["lesson"]["status"] == "candidate" and len(r["support_runs"]) >= AUTO_VALIDATE_AFTER:
                        r["lesson"]["status"] = "validated"
                    self._write(rows)
                    return Lesson.model_validate(r["lesson"])
            rows.append({"lesson": lesson.model_dump(), "vec": v.round(6).tolist(),
                         "support_runs": [lesson.source_run] if lesson.source_run else []})
            self._write(rows)
            return lesson

    def search(self, query: str, k: int, include_candidates: bool = True) -> list[Lesson]:
        rows = [r for r in self._load() if r["lesson"]["status"] != "rejected"
                and (include_candidates or r["lesson"]["status"] == "validated")]
        if not rows:
            return []
        q = self.embed(query)
        sims = np.asarray([r["vec"] for r in rows]) @ q
        out = []
        for i in np.argsort(-sims)[:k]:
            if sims[i] <= 0.05:
                continue
            les = Lesson.model_validate(rows[i]["lesson"])
            les.score = round(float(sims[i]), 4)
            out.append(les)
        return out

    def all(self) -> list[Lesson]:
        return [Lesson.model_validate(r["lesson"]) for r in self._load()]

    def set_status(self, lesson_id: str, status: str) -> Lesson:
        with self._lock:
            rows = self._load()
            for r in rows:
                if r["lesson"]["lesson_id"] == lesson_id:
                    r["lesson"]["status"] = status
                    self._write(rows)
                    return Lesson.model_validate(r["lesson"])
        raise KeyError(lesson_id)


class ChromaMemory(MemoryStore):
    def __init__(self, cfg: MemoryConfig, root: Path):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(root / "chroma_db"))
        self.col = self.client.get_or_create_collection("lessons", metadata={"hnsw:space": "cosine"})
        self.embed = Embedder(cfg)

    def add(self, lesson: Lesson) -> Lesson:
        v = self.embed(lesson.text).tolist()
        hit = self.col.query(query_embeddings=[v], n_results=1) if self.col.count() else None
        if hit and hit["ids"][0] and 1 - hit["distances"][0][0] >= DEDUP_SIM:
            meta = hit["metadatas"][0][0]
            les = Lesson.model_validate_json(meta["json"])
            runs = set(json.loads(meta.get("support_runs", "[]"))) | {lesson.source_run}
            runs.discard(None)
            if les.status == "candidate" and len(runs) >= AUTO_VALIDATE_AFTER:
                les.status = "validated"
            self.col.update(ids=[les.lesson_id], metadatas=[{"json": les.model_dump_json(), "status": les.status,
                                                             "support_runs": json.dumps(sorted(runs))}])
            return les
        self.col.add(ids=[lesson.lesson_id], embeddings=[v], documents=[lesson.text],
                     metadatas=[{"json": lesson.model_dump_json(), "status": lesson.status,
                                 "support_runs": json.dumps([lesson.source_run] if lesson.source_run else [])}])
        return lesson

    def search(self, query: str, k: int, include_candidates: bool = True) -> list[Lesson]:
        if not self.col.count():
            return []
        where = None if include_candidates else {"status": "validated"}
        res = self.col.query(query_embeddings=[self.embed(query).tolist()], n_results=min(k, self.col.count()), where=where)
        out = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            les = Lesson.model_validate_json(meta["json"])
            if les.status == "rejected":
                continue
            les.score = round(1 - dist, 4)
            out.append(les)
        return out

    def all(self) -> list[Lesson]:
        got = self.col.get()
        return [Lesson.model_validate_json(m["json"]) for m in got["metadatas"]]

    def set_status(self, lesson_id: str, status: str) -> Lesson:
        got = self.col.get(ids=[lesson_id])
        if not got["ids"]:
            raise KeyError(lesson_id)
        les = Lesson.model_validate_json(got["metadatas"][0]["json"])
        les.status = status
        meta = dict(got["metadatas"][0], json=les.model_dump_json(), status=status)
        self.col.update(ids=[lesson_id], metadatas=[meta])
        return les


def make_memory(cfg: MemoryConfig, root: Path) -> MemoryStore:
    if cfg.backend == "chroma":
        return ChromaMemory(cfg, root)
    return LocalMemory(cfg, root)
