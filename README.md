# bioagent

A modular bioinformatics agent built from `architecture_specification.md`. It keeps reasoning (a LangGraph
**cognitive graph**) separate from compute (a **compute graph** of subprocess, Snakemake or Nextflow jobs). State is
typed, judgment is explicit and backed by evidence, there are human-in-the-loop approvals and a tamper-evident
provenance log, and one config switches between cloud and air-gapped local operation.

```
retrieve → plan (hypotheses + Tree-of-Thoughts) → design → validate ─┬→ approval (HITL) ─┐
                                                       ▲             └──────────────────→ execute → [suspend] await_job
                                                       │                                                     │
                       anomaly ◄── yes ── qc (deterministic findings + binary LLM judgment) ◄────────────────┘
                          │                 │ no
                          └─→ design / next stage → … → interpret → reflect (memory) → report
```

## Quick start (no model, no network)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q                      # 24 tests: graph, compute, QC statistics, HITL, resume-after-restart, API
bioagent demo                  # synthetic RNA-seq with a planted batch effect, mock LLM
bioagent demo --fully-confounded --out examples/fc
bioagent --config configs/mock.yaml serve   # dashboard at http://127.0.0.1:3000
```

The demo uses a **deterministic mock LLM**, so it tests the machinery (routing, typed state, compute, QC, anomaly
handling, memory, provenance) and not the quality of the model's reasoning. Each synthetic dataset comes with
ground truth, and every run is scored against it (`## Evaluation` in the report).

## Real models

| Config | Premium (judgment, planning, interpretation) | Budget (parsing, formatting) | Compute |
|---|---|---|---|
| `configs/local.yaml` | Ollama `qwen2.5:32b` | Ollama `qwen2.5:7b` | local subprocess / Snakemake |
| `configs/cloud.yaml` | Anthropic | DeepSeek (OpenAI-compatible) | Nextflow on AWS Batch / K8s |

```bash
ollama serve & ollama pull qwen2.5:32b && ollama pull qwen2.5:7b
bioagent --config configs/local.yaml run examples/synthetic/task.yaml
```

`mode: local` is enforced, not advisory. The config refuses to load if any LLM endpoint, embedding endpoint or
telemetry sink is not on a loopback or private address, and OpenAPI tools that point at external hosts are blocked.

If the project lives on a network drive or a synced folder (Dropbox, iCloud, OneDrive), keep the checkpoint
database on local disk with `export BIOAGENT_STATE_DIR=~/.bioagent/state`. SQLite WAL mode, which LangGraph uses,
fails on many such filesystems.

## Task file

```yaml
goal: Identify genes differentially expressed between treated and control, controlling for technical artefacts.
analysis_type: rnaseq_de          # rnaseq_de | rnaseq_fastq | genome_assembly | custom
inputs: {counts: data/counts.tsv, metadata: data/metadata.tsv}
condition_column: condition
contrast: [treated, control]
ground_truth: {de_truth: data/truth.tsv}   # optional, used only for scoring, never shown to the LLM
```

## Adding tools (no code changes)

| Drop in | Becomes |
|---|---|
| `tools/<name>/tool.yaml` | CLI tool: command template, conditional args, container/conda, declared outputs, QC parser |
| `tools/python/<name>.py` with a literal `TOOL_SPEC = {...}` | Python tool (spec read with `ast.literal_eval`, never imported) |
| `tools/openapi/<name>.openapi.yaml` | one tool per `operationId` |
| `workflows/<id>/workflow.yaml` + `Snakefile` / `main.nf` | workflow runnable by Snakemake (local) or Nextflow (cloud) |

`bioagent tools` prints the registry exactly as the LLM sees it, including which tools are unavailable and why.
To give a tool's outputs a deterministic QC judgment, register an evaluator in `bioagent/qc.py`.

## Layout

```
bioagent/
  config.py         settings + cloud/local toggle + air-gap enforcement
  state.py          typed Blackboard and every LLM output schema (Pydantic)
  transitions.py    admissible phase transitions + evidence-based admissible anomaly resolutions
  graph/            LangGraph nodes, prompts, builder (SQLite checkpoints)
  llm/              router (tiering, escalation, spend cap, schema repair), HTTP providers, mock
  compute.py        materializer + job manager (local / snakemake / nextflow)
  sandbox.py        static checks, rlimits, network isolation, docker
  registry.py       runtime tool/workflow discovery
  qc.py             deterministic QC evaluators (fastp, STAR, featureCounts, QUAST, confounders, DE)
  memory.py         reflection memory with candidate → validated lifecycle (local or Chroma)
  provenance.py     hash-chained JSONL + event bus
  observability.py  Phoenix / W&B exporters, PRM step-dataset export
  api/              FastAPI: REST, WebSocket trace, HITL approvals, webhooks, dashboard
tools/  workflows/  configs/  tests/
```

## Run artefacts (`runs/<run_id>/`)

`report.md` / `report.pdf`, `blackboard.json` (full typed final state), `provenance.jsonl` (hash-chained; check
with `bioagent verify <run_id>`), `llm_calls.jsonl` (every prompt and response), `jobs/*/` (commands, logs,
outputs), and `prm_steps.jsonl` (`bioagent prm-export <run_id>`).

See `DESIGN.md` for the decisions behind this, where the build departs from the spec, and the known limits.
