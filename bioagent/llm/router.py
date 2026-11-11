"""LLM router: task classification, tier selection, cost control, typed output.

Routing is deterministic by design. Each node declares a TaskClass; the
router maps it to a tier (premium/budget) from config, then applies two
dynamic rules:
  * escalation  - a budget-tier call that fails schema validation
                  `escalate_after_failures` times is retried on premium;
  * spend cap   - once the run's spend reaches `run_budget_usd`, non-judgment
                  tasks are forced to budget and judgment tasks raise.
A learned classifier was deliberately not used: routing decisions must be
reproducible to compare runs, and the node already knows what it is asking.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig, ModelSpec
from ..provenance import Provenance, sha256_text
from ..state import CostLedger, now
from .providers import Provider, ProviderError, make_provider

T = TypeVar("T", bound=BaseModel)


class TaskClass(str, Enum):
    JUDGMENT = "judgment"
    HYPOTHESIS = "hypothesis"
    PLANNING = "planning"
    DESIGN = "design"
    ANOMALY = "anomaly"
    INTERPRETATION = "interpretation"
    REFLECTION = "reflection"
    SCORING = "scoring"
    PARSING = "parsing"
    FORMATTING = "formatting"
    SUMMARIZATION = "summarization"


PROTECTED = {TaskClass.JUDGMENT, TaskClass.ANOMALY}


class BudgetExceeded(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("unterminated JSON in response")


class LLMRouter:
    def __init__(self, cfg: LLMConfig, providers: dict[str, Provider] | None = None):
        self.cfg = cfg
        self.providers = providers or {"premium": make_provider(cfg.premium), "budget": make_provider(cfg.budget)}

    def tier_for(self, task: TaskClass, ledger: CostLedger) -> str:
        tier = "budget" if task.value in self.cfg.budget_tasks else "premium"
        if self.cfg.run_budget_usd > 0 and ledger.usd >= self.cfg.run_budget_usd:
            if task in PROTECTED:
                raise BudgetExceeded(
                    f"run spend ${ledger.usd:.2f} >= cap ${self.cfg.run_budget_usd:.2f}; refusing {task.value} call"
                )
            tier = "budget"
        return tier

    def _spec(self, tier: str) -> ModelSpec:
        return getattr(self.cfg, tier)

    def call(
        self,
        task: TaskClass,
        system: str,
        prompt: str,
        schema: type[T] | None,
        ledger: CostLedger,
        prov: Provenance | None = None,
        ctx: dict | None = None,
        node: str | None = None,
    ) -> T | str:
        ctx = dict(ctx or {})
        ctx["task_class"] = task.value
        ctx["schema"] = schema.__name__ if schema else None
        tier = self.tier_for(task, ledger)
        json_schema = schema.model_json_schema() if schema else None
        sys_prompt = system
        if schema is not None:
            sys_prompt += (
                "\n\nRespond with a single JSON object only, no prose, conforming to this JSON schema:\n"
                + json.dumps(json_schema)
            )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        failures = 0
        attempts = 0
        max_attempts = 1 + self.cfg.max_repair_attempts + (self.cfg.escalate_after_failures if tier == "budget" else 0)
        last_err = ""
        while attempts < max_attempts:
            attempts += 1
            provider = self.providers[tier]
            try:
                resp = provider.complete(sys_prompt, messages, json_schema, ctx)
            except ProviderError as e:
                last_err = str(e)
                self._record(prov, node, task, tier, provider, None, sys_prompt, messages, error=last_err)
                if tier == "budget":
                    tier, failures = "premium", 0
                    ledger.escalations += 1
                    continue
                raise
            self._account(ledger, tier, resp)
            if schema is None:
                self._record(prov, node, task, tier, provider, resp, sys_prompt, messages)
                return resp.text
            try:
                obj = schema.model_validate(extract_json(resp.text))
                self._record(prov, node, task, tier, provider, resp, sys_prompt, messages)
                return obj
            except (ValueError, ValidationError) as e:
                failures += 1
                ledger.schema_failures += 1
                last_err = str(e)[:1500]
                self._record(prov, node, task, tier, provider, resp, sys_prompt, messages, error=last_err)
                messages = messages + [
                    {"role": "assistant", "content": resp.text[:4000]},
                    {
                        "role": "user",
                        "content": f"That response failed validation:\n{last_err}\nReturn corrected JSON only.",
                    },
                ]
                if tier == "budget" and failures >= self.cfg.escalate_after_failures:
                    tier, failures = "premium", 0
                    ledger.escalations += 1
        raise StructuredOutputError(f"{task.value}: no valid {schema.__name__ if schema else 'text'} after {attempts} attempts: {last_err}")

    def _account(self, ledger: CostLedger, tier: str, resp) -> None:
        spec = self._spec(tier)
        ledger.tokens_in += resp.tokens_in
        ledger.tokens_out += resp.tokens_out
        ledger.usd += resp.tokens_in / 1e6 * spec.cost_in_per_mtok + resp.tokens_out / 1e6 * spec.cost_out_per_mtok
        ledger.usd = round(ledger.usd, 6)
        ledger.calls_by_tier[tier] = ledger.calls_by_tier.get(tier, 0) + 1

    def _record(self, prov, node, task, tier, provider, resp, system, messages, error=None):
        if prov is None:
            return
        prompt_text = system + "\n" + json.dumps(messages)
        prov.log(
            "llm_call",
            node=node,
            task_class=task.value,
            tier=tier,
            model=provider.name,
            prompt_sha256=sha256_text(prompt_text),
            response_sha256=sha256_text(resp.text) if resp else None,
            tokens_in=resp.tokens_in if resp else 0,
            tokens_out=resp.tokens_out if resp else 0,
            latency_s=round(resp.latency_s, 3) if resp else None,
            error=error,
        )
        # Full text goes to a side file: provenance stays small, research keeps everything.
        with (Path(prov.path).parent / "llm_calls.jsonl").open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": now(),
                        "node": node,
                        "task_class": task.value,
                        "tier": tier,
                        "model": provider.name,
                        "system": system,
                        "messages": messages,
                        "response": resp.text if resp else None,
                        "error": error,
                    }
                )
                + "\n"
            )
