"""Hash-chained JSONL provenance plus an in-process event bus.

Every record carries `prev` (hash of the previous record) and `hash`, so a
run's log can be verified for tampering or truncation with `verify()`.
Subscribers (WebSocket clients, Phoenix/W&B exporters) receive the same
records as they are written.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from .state import now

Subscriber = Callable[[dict], None]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def sha256_file(path: str | Path, limit_bytes: int | None = 2 * 1024**3) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    read = 0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_bytes and read >= limit_bytes:
                return h.hexdigest() + ":truncated"
    return h.hexdigest()


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Subscriber]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str, fn: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.setdefault(run_id, []).append(fn)
            self._subs.setdefault("*", [])

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs.get(run_id, []):
                    self._subs[run_id].remove(fn)

        return unsubscribe

    def publish(self, run_id: str, record: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(run_id, [])) + list(self._subs.get("*", []))
        for fn in subs:
            try:
                fn(record)
            except Exception:  # a broken subscriber must never break a run
                pass


BUS = EventBus()


class Provenance:
    """One instance per run; safe to recreate on resume (continues the chain)."""

    def __init__(self, run_dir: str | Path, run_id: str, bus: EventBus = BUS) -> None:
        self.run_id = run_id
        self.path = Path(run_dir) / "provenance.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bus = bus
        self._lock = threading.Lock()
        self._seq, self._prev = self._tail()

    def _tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, "genesis"
        last = None
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return 0, "genesis"
        rec = json.loads(last)
        return rec["seq"] + 1, rec["hash"]

    def log(self, event: str, **data: Any) -> dict:
        with self._lock:
            rec = {"seq": self._seq, "ts": now(), "run_id": self.run_id, "event": event, **data, "prev": self._prev}
            body = json.dumps(rec, sort_keys=True, default=str)
            rec["hash"] = sha256_text(body)
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            self._seq += 1
            self._prev = rec["hash"]
        self.bus.publish(self.run_id, rec)
        return rec

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        p = Path(path)
        if p.is_dir():
            p = p / "provenance.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    @staticmethod
    def verify(path: str | Path) -> tuple[bool, str]:
        prev = "genesis"
        for i, rec in enumerate(Provenance.read(path)):
            h = rec.pop("hash")
            if rec["prev"] != prev:
                return False, f"chain broken at seq {rec['seq']} (line {i})"
            if sha256_text(json.dumps(rec, sort_keys=True, default=str)) != h:
                return False, f"record {rec['seq']} modified"
            prev = h
        return True, "ok"
