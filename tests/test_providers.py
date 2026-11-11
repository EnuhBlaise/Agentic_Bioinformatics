"""Request/response handling for HTTP providers against canned API payloads (no network)."""
import copy

import httpx
import pytest

from bioagent.config import ModelSpec
from bioagent.llm.providers import ProviderError, make_provider


class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._d, self.text = status, data, str(data)

    def json(self):
        return self._d


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def install(responses):
        it = iter(responses)

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "headers": headers, "json": copy.deepcopy(json)})
            return next(it)
        monkeypatch.setattr(httpx, "post", fake_post)
        return calls
    return install


def test_anthropic(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    calls = capture([FakeResp(200, {"content": [{"type": "text", "text": "{\"a\":1}"}],
                                    "usage": {"input_tokens": 11, "output_tokens": 3}})])
    r = make_provider(ModelSpec(provider="anthropic", model="m")).complete("sys", [{"role": "user", "content": "hi"}], {}, {})
    assert r.text == '{"a":1}' and r.tokens_in == 11
    assert calls[0]["url"].endswith("/v1/messages") and calls[0]["headers"]["x-api-key"] == "k"
    assert calls[0]["json"]["system"] == "sys"


def test_openai_compatible_falls_back_without_response_format(capture):
    calls = capture([FakeResp(400, {"error": "response_format not supported"}),
                     FakeResp(200, {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})])
    r = make_provider(ModelSpec(provider="openai_compatible", model="m", base_url="http://localhost:8000/v1")) \
        .complete("s", [{"role": "user", "content": "x"}], {"type": "object"}, {})
    assert r.text == "{}" and "response_format" in calls[0]["json"] and "response_format" not in calls[1]["json"]
    assert calls[0]["json"]["messages"][0] == {"role": "system", "content": "s"}


def test_ollama_schema_constrained(capture):
    calls = capture([FakeResp(200, {"message": {"content": "{\"x\":2}"}, "prompt_eval_count": 7, "eval_count": 2})])
    r = make_provider(ModelSpec(provider="ollama", model="qwen2.5:7b")).complete("s", [], {"type": "object"}, {})
    assert r.tokens_out == 2 and calls[0]["json"]["format"] == {"type": "object"} and calls[0]["url"].endswith("/api/chat")


def test_missing_key_is_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        make_provider(ModelSpec(provider="openai", model="m")).complete("s", [], None, {})
