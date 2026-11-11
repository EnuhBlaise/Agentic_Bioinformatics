"""Synthetic RNA-seq benchmark with known ground truth.

Negative-binomial counts with:
  * `n_de` true DE genes (condition effect),
  * a strong batch effect on `batch_frac` of genes,
  * batch partially confounded with condition (4/2 vs 2/4 by default),
so an analysis that ignores batch produces false positives and one that
models it does not. With `fully_confounded=True` batch == condition, which no
statistical adjustment can fix; the correct agent behaviour is to say so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def make_dataset(outdir: str | Path, n_genes: int = 3000, n_per_group: int = 6, n_de: int = 200,
                 batch_frac: float = 0.3, batch_sd: float = 1.2, fully_confounded: bool = False,
                 dispersion: float = 0.08, seed: int = 7) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    n = 2 * n_per_group
    cond = np.array(["control"] * n_per_group + ["treated"] * n_per_group)
    if fully_confounded:
        batch = np.where(cond == "control", "b1", "b2")
    else:
        k = int(round(n_per_group * 2 / 3))
        batch = np.array(["b1"] * k + ["b2"] * (n_per_group - k) + ["b1"] * (n_per_group - k) + ["b2"] * k)
    samples = [f"S{i+1:02d}" for i in range(n)]
    genes = [f"GENE{i+1:05d}" for i in range(n_genes)]

    base = np.exp(rng.normal(5.0, 1.6, n_genes))
    lfc = np.zeros(n_genes)
    de_idx = rng.choice(n_genes, n_de, replace=False)
    lfc[de_idx] = rng.choice([-1, 1], n_de) * rng.uniform(1.0, 2.5, n_de)
    blfc = np.zeros(n_genes)
    b_idx = rng.choice(n_genes, int(batch_frac * n_genes), replace=False)
    blfc[b_idx] = rng.normal(0, batch_sd, len(b_idx))
    size = rng.uniform(0.7, 1.4, n)

    mu = base[:, None] * size[None, :] * 2 ** (lfc[:, None] * (cond == "treated")[None, :]) \
        * 2 ** (blfc[:, None] * (batch == "b2")[None, :])
    r = 1 / dispersion
    counts = rng.negative_binomial(r, r / (r + mu))

    cpath, mpath, tpath = out / "counts.tsv", out / "metadata.tsv", out / "truth.tsv"
    pd.DataFrame(counts, index=pd.Index(genes, name="gene"), columns=samples).to_csv(cpath, sep="\t")
    pd.DataFrame({"sample": samples, "condition": cond, "batch": batch,
                  "rin": np.round(rng.uniform(7.5, 9.5, n), 1)}).to_csv(mpath, sep="\t", index=False)
    pd.DataFrame({"gene": genes, "true_log2fc": lfc, "is_de": lfc != 0, "batch_log2fc": blfc}).to_csv(tpath, sep="\t", index=False)

    task = {
        "goal": "Identify genes differentially expressed between treated and control samples, "
                "controlling for technical artefacts.",
        "analysis_type": "rnaseq_de",
        "organism": "synthetic",
        "inputs": {"counts": str(cpath.resolve()), "metadata": str(mpath.resolve())},
        "condition_column": "condition",
        "contrast": ["treated", "control"],
        "ground_truth": {"de_truth": str(tpath.resolve())},
    }
    (out / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    return {"counts": str(cpath), "metadata": str(mpath), "truth": str(tpath), "task": str(out / "task.yaml")}


def evaluate_de(results_tsv: str, truth_tsv: str, alpha: float = 0.05) -> dict:
    res = pd.read_csv(results_tsv, sep="\t", index_col=0)
    truth = pd.read_csv(truth_tsv, sep="\t", index_col=0)
    called = set(res.index[res["padj"] < alpha])
    true = set(truth.index[truth["is_de"]])
    tp = len(called & true)
    fp = len(called - true)
    return {"called": len(called), "true_de": len(true), "tp": tp, "fp": fp,
            "fdr": round(fp / max(len(called), 1), 4), "recall": round(tp / max(len(true), 1), 4)}


if __name__ == "__main__":
    import sys

    print(json.dumps(make_dataset(sys.argv[1] if len(sys.argv) > 1 else "examples/synthetic"), indent=2))
