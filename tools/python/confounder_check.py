#!/usr/bin/env python3
"""Batch-effect and confounder check for a gene x sample count matrix.

Computes, on log2-CPM of the most variable genes:
  * PCA and per-PC association (R^2, p) with every metadata covariate
  * variance-weighted R^2 per covariate (share of top-gene variance it explains)
  * condition x covariate confounding (Cramer's V and design-matrix rank)
  * outlier samples (robust z of median inter-sample correlation)
  * library-size imbalance
Writes report.json and pca.tsv. The flags are deterministic so the judgment
layer has evidence that does not depend on an LLM reading a plot.
"""
TOOL_SPEC = {
    "id": "confounder_check",
    "name": "PCA batch-effect / confounder check",
    "category": "qc",
    "description": "PCA of log-CPM counts; tests association of PCs with metadata covariates, "
                   "detects batch effects, condition-batch confounding, outlier samples. Run BEFORE differential expression.",
    "inputs": {
        "counts": {"type": "path", "required": True, "description": "TSV, genes x samples, first column gene id"},
        "metadata": {"type": "path", "required": True, "description": "TSV, one row per sample"},
        "condition": {"type": "str", "required": True, "description": "metadata column of biological interest"},
        "sample_col": {"type": "str", "default": "sample"},
        "covariates": {"type": "list", "description": "columns to test; default: all except sample/condition"},
        "exclude_samples": {"type": "list", "description": "samples to drop before analysis"},
        "top_genes": {"type": "int", "default": 500},
        "n_pcs": {"type": "int", "default": 5},
    },
    "outputs": {"report": "{outdir}/report.json", "pca": "{outdir}/pca.tsv"},
    "qc_parser": "confounder_check",
}

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(counts_path, meta_path, sample_col):
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    meta = pd.read_csv(meta_path, sep="\t", dtype=str)
    if sample_col not in meta.columns:
        raise SystemExit(f"sample column '{sample_col}' not in metadata columns {list(meta.columns)}")
    meta = meta.set_index(sample_col)
    missing = [s for s in counts.columns if s not in meta.index]
    if missing:
        raise SystemExit(f"samples in counts but not metadata: {missing}")
    meta = meta.loc[list(counts.columns)]
    return counts, meta


def log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    libsize = counts.sum(axis=0)
    return np.log2((counts + 0.5) / (libsize + 1.0) * 1e6)


def is_numeric(s: pd.Series) -> bool:
    try:
        v = pd.to_numeric(s)
        return v.nunique() > 5
    except (ValueError, TypeError):
        return False


def association(pc: np.ndarray, cov: pd.Series):
    if is_numeric(cov):
        x = pd.to_numeric(cov).to_numpy(float)
        r, p = stats.pearsonr(x, pc)
        return float(r * r), float(p), "numeric"
    groups = [pc[(cov == lvl).to_numpy()] for lvl in cov.unique()]
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        return 0.0, 1.0, "categorical"
    grand = pc.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((pc - grand) ** 2).sum()
    r2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
    k, n = len(groups), len(pc)
    if n - k <= 0 or r2 >= 1:
        return r2, 0.0 if r2 >= 1 else 1.0, "categorical"
    f = (r2 / (k - 1)) / ((1 - r2) / (n - k))
    return r2, float(stats.f.sf(f, k - 1, n - k)), "categorical"


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    if min(tab.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def design_rank_deficient(meta: pd.DataFrame, cols: list[str]) -> bool:
    X = pd.get_dummies(meta[cols].astype(str), drop_first=True).astype(float)
    X.insert(0, "intercept", 1.0)
    return int(np.linalg.matrix_rank(X.to_numpy())) < X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--sample_col", default="sample")
    ap.add_argument("--covariates", nargs="*", default=None)
    ap.add_argument("--exclude_samples", nargs="*", default=[])
    ap.add_argument("--top_genes", type=int, default=500)
    ap.add_argument("--n_pcs", type=int, default=5)
    ap.add_argument("--outdir", required=True)
    # thresholds (exposed so they can be studied, not because they are universal)
    ap.add_argument("--pc_var_min", type=float, default=0.10)
    ap.add_argument("--r2_min", type=float, default=0.5)
    ap.add_argument("--weighted_r2_min", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    counts, meta = load(a.counts, a.metadata, a.sample_col)
    keep = [c for c in counts.columns if c not in set(a.exclude_samples)]
    counts, meta = counts[keep], meta.loc[keep]
    if a.condition not in meta.columns:
        raise SystemExit(f"condition column '{a.condition}' not in metadata {list(meta.columns)}")
    covs = a.covariates or [c for c in meta.columns if c != a.condition]
    covs = [c for c in covs if c in meta.columns and meta[c].nunique() > 1]

    expressed = counts[(counts.sum(axis=1) > 0)]
    lc = log_cpm(expressed)
    top = lc.loc[lc.var(axis=1).sort_values(ascending=False).index[: a.top_genes]]
    X = (top.T - top.T.mean(axis=0)).to_numpy()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2 / (S**2).sum()
    n_pcs = min(a.n_pcs, len(S))
    pcs = U[:, :n_pcs] * S[:n_pcs]
    pca_df = pd.DataFrame(pcs, index=counts.columns, columns=[f"PC{i+1}" for i in range(n_pcs)])
    pca_df = pca_df.join(meta)
    pca_df.to_csv(out / "pca.tsv", sep="\t", index_label="sample")

    assoc = {}
    for c in [a.condition, *covs]:
        rows = []
        for i in range(n_pcs):
            r2, p, kind = association(pcs[:, i], meta[c])
            rows.append({"pc": f"PC{i+1}", "var_explained": round(float(var[i]), 4), "r2": round(r2, 4), "p": p, "type": kind})
        weighted = float(sum(r["var_explained"] * r["r2"] for r in rows) / sum(r["var_explained"] for r in rows))
        assoc[c] = {"per_pc": rows, "weighted_r2": round(weighted, 4)}

    # Bonferroni over every (covariate, PC) test: with ~12 samples, chance
    # correlations of a random covariate with some PC are common.
    n_tests = max(1, len(covs) * n_pcs)
    alpha_adj = a.alpha / n_tests
    batch_hits = []
    for c in covs:
        rows = assoc[c]["per_pc"]
        sig = [r for r in rows if r["p"] < alpha_adj]
        strong = [r for r in sig if r["var_explained"] >= a.pc_var_min and r["r2"] >= a.r2_min]
        if strong or (sig and assoc[c]["weighted_r2"] >= a.weighted_r2_min):
            batch_hits.append({"covariate": c, "weighted_r2": assoc[c]["weighted_r2"],
                               "strong_pcs": [r["pc"] for r in strong], "min_p": min(r["p"] for r in rows)})

    confounding = []
    for c in covs:
        if is_numeric(meta[c]):
            continue
        v = cramers_v(meta[a.condition], meta[c])
        confounding.append({"covariate": c, "cramers_v": round(v, 4),
                            "fully_confounded": design_rank_deficient(meta, [c, a.condition]),
                            "table": pd.crosstab(meta[c], meta[a.condition]).to_dict()})

    # Outliers: median correlation of each sample to all others (top genes),
    # lower-tail robust z. Robust to multi-cluster structure (batches, groups),
    # unlike distance-from-centroid.
    cor = np.corrcoef(top.T.to_numpy())
    np.fill_diagonal(cor, np.nan)
    medcor = np.nanmedian(cor, axis=1)
    m = np.median(medcor)
    mad = np.median(np.abs(medcor - m)) * 1.4826 or 1e-9
    rz = (medcor - m) / mad
    outliers = []
    if len(rz) >= 6:
        outliers = [{"sample": s, "robust_z": round(float(z), 2), "median_corr": round(float(mc), 4)}
                    for s, z, mc in zip(counts.columns, rz, medcor) if z < -3.5]

    libsize = counts.sum(axis=0)
    report = {
        "n_genes": int(counts.shape[0]), "n_expressed": int(expressed.shape[0]), "n_samples": int(counts.shape[1]),
        "condition": a.condition, "covariates_tested": covs,
        "pca_variance_explained": [round(float(v), 4) for v in var[:n_pcs]],
        "associations": assoc,
        "batch_effect": {"flagged": bool(batch_hits), "hits": batch_hits},
        "confounding": confounding,
        "outliers": outliers,
        "library_size": {"min": int(libsize.min()), "max": int(libsize.max()),
                         "ratio": round(float(libsize.max() / max(libsize.min(), 1)), 3)},
        "thresholds": {"bonferroni_alpha": alpha_adj, "pc_var_min": a.pc_var_min, "r2_min": a.r2_min,
                       "weighted_r2_min": a.weighted_r2_min, "alpha": a.alpha},
    }
    hits = [h["covariate"] for h in batch_hits]
    aliased = [c["covariate"] for c in confounding if c["fully_confounded"]]
    adjustable = [h for h in hits if h not in aliased]
    report["recommended_design"] = "~ " + " + ".join([*adjustable, a.condition])
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"batch_effect={bool(batch_hits)} hits={hits} fully_confounded={aliased} outliers={[o['sample'] for o in outliers]}")


if __name__ == "__main__":
    main()
