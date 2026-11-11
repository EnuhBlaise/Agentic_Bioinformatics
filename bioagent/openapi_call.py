"""Executor for OpenAPI-derived tools (run as a compute step so it is logged and sandboxed like any other)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="GET")
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    params = json.loads(a.params)
    url = a.url
    for k in list(params):
        if "{" + k + "}" in url:
            url = url.replace("{" + k + "}", str(params.pop(k)))
    if re.search(r"\{[^}]+\}", url):
        raise SystemExit(f"unfilled path parameters in {url}")
    if a.method == "GET":
        r = httpx.get(url, params=params, timeout=60)
    else:
        r = httpx.post(url, json=params, timeout=60)
    r.raise_for_status()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(json.dumps(r.json(), indent=2))
    except ValueError:
        out.write_text(json.dumps({"text": r.text}))
    print(f"{a.method} {r.url} -> {r.status_code}, {len(r.content)} bytes")


if __name__ == "__main__":
    main()
