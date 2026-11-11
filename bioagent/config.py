"""Configuration and the cloud/local environment toggle.

A single YAML file selects every backend. `mode: local` turns on air-gap
enforcement: any LLM endpoint, OpenAPI server or telemetry sink that is not on
a loopback/private address is refused at startup rather than silently used.
"""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelSpec(BaseModel):
    provider: Literal["anthropic", "openai", "openai_compatible", "ollama", "mock"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_s: float = 180.0
    # USD per 1M tokens; used by the router's cost ledger. Local models default to 0.
    cost_in_per_mtok: float = 0.0
    cost_out_per_mtok: float = 0.0


class LLMConfig(BaseModel):
    premium: ModelSpec
    budget: ModelSpec
    # Task classes that may use the budget tier. Everything else goes premium.
    budget_tasks: list[str] = Field(
        default_factory=lambda: ["parsing", "formatting", "summarization"]
    )
    # Escalate a budget call to premium after this many schema-validation failures.
    escalate_after_failures: int = 2
    max_repair_attempts: int = 2
    # Hard spend ceiling per run (USD). When exceeded, non-judgment tasks are
    # downgraded to budget and the run fails if even judgment would exceed it.
    run_budget_usd: float = 10.0  # <= 0 disables the cap


class ComputeConfig(BaseModel):
    backend: Literal["local", "snakemake", "nextflow"] = "local"
    cores: int = 4
    memory_gb: int = 8
    job_timeout_s: int = 6 * 3600
    use_containers: bool = False  # docker for tool.yaml containers when available
    nextflow_profile: str = "awsbatch"  # or "k8s"
    nextflow_work_dir: str | None = None  # e.g. s3://bucket/work
    webhook_base_url: str | None = None  # where Nextflow -with-weblog posts events
    sandbox_network: bool = False  # allow network inside sandboxed code


class MemoryConfig(BaseModel):
    backend: Literal["local", "chroma"] = "local"
    path: str = "./state/memory"
    top_k: int = 5
    # Lessons start as "candidate". Only validated lessons are injected as
    # rules; candidates are shown as unverified hints. See DESIGN.md.
    inject_candidates: bool = True
    embedding: Literal["hashing", "ollama"] = "hashing"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"


class ObservabilityConfig(BaseModel):
    phoenix: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    langsmith: bool = False
    wandb: bool = False
    wandb_project: str = "bioagent-prm"


class HITLConfig(BaseModel):
    # always: every execution needs a human. risky: only flagged/LLM-written code.
    # never: fully autonomous (use for benchmarks only).
    approval_policy: Literal["always", "risky", "never"] = "risky"


class GraphConfig(BaseModel):
    tot_branches: int = 3
    max_cycles: int = 40
    max_design_revisions: int = 3
    max_job_retries: int = 2
    # How deterministic checks and LLM judgment combine:
    #  conservative: anomaly if either flags it
    #  rule: deterministic rule wins, LLM recorded only
    #  llm: LLM wins, rule recorded only
    judgment_combination: Literal["conservative", "rule", "llm"] = "conservative"


class Settings(BaseModel):
    mode: Literal["local", "cloud"] = "local"
    workspace: str = "./runs"
    state_dir: str = "./state"
    tools_dir: str = "./tools"
    workflows_dir: str = "./workflows"
    llm: LLMConfig
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    # Path the config was loaded from; relative paths resolve against its parent.
    config_dir: str = "."

    @model_validator(mode="after")
    def _airgap(self) -> "Settings":
        if self.mode != "local":
            return self
        problems = []
        for tier in ("premium", "budget"):
            spec: ModelSpec = getattr(self.llm, tier)
            if spec.provider in ("anthropic", "openai"):
                problems.append(f"llm.{tier} uses hosted provider '{spec.provider}'")
            if spec.base_url and not is_local_url(spec.base_url):
                problems.append(f"llm.{tier}.base_url {spec.base_url} is not local")
        if self.observability.langsmith or self.observability.wandb:
            problems.append("LangSmith/W&B telemetry leaves the machine; use phoenix in local mode")
        if self.observability.phoenix and not is_local_url(self.observability.phoenix_endpoint):
            problems.append("phoenix_endpoint is not local")
        if self.compute.backend == "nextflow" and self.compute.nextflow_profile in ("awsbatch", "k8s"):
            problems.append("nextflow cloud profile selected in local mode")
        if self.memory.embedding == "ollama" and not is_local_url(self.memory.embedding_base_url):
            problems.append("embedding endpoint is not local")
        if problems:
            raise ValueError("mode=local (air-gapped) violations: " + "; ".join(problems))
        return self

    def resolve(self, p: str) -> Path:
        path = Path(os.path.expanduser(p))
        return path if path.is_absolute() else (Path(self.config_dir) / path).resolve()


def is_local_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def load_settings(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Settings:
    path = path or os.environ.get("BIOAGENT_CONFIG", "configs/local.yaml")
    p = Path(path).resolve()
    data = yaml.safe_load(p.read_text()) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    # Environment overrides, e.g. to keep the SQLite checkpoint DB off a network/FUSE/synced folder
    # (LangGraph uses SQLite WAL mode, which needs shared-memory files those filesystems often lack).
    if os.environ.get("BIOAGENT_STATE_DIR"):
        sd = os.environ["BIOAGENT_STATE_DIR"]
        data["state_dir"] = sd
        data.setdefault("memory", {})["path"] = str(Path(sd) / "memory")
    if os.environ.get("BIOAGENT_WORKSPACE"):
        data["workspace"] = os.environ["BIOAGENT_WORKSPACE"]
    data.setdefault("config_dir", str(p.parent.parent if p.parent.name == "configs" else p.parent))
    return Settings.model_validate(data)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
