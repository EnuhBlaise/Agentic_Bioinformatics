#!/usr/bin/env python3
"""Differential expression with arbitrary covariates in the design.

method=pydeseq2  PyDESeq2 (negative binomial GLM, DESeq2 port) if installed
method=ebayes    gene-wise OLS on log2-CPM with empirical-Bayes variance
                 moderation (limma-style constant prior, no voom weights).
                 Adequate for screening and for the demo; use DESeq2/edgeR/
                 limma-voom for publication.
method=auto      pydeseq2 if importable, else ebayes.
"""
TOOL_SPEC = {
    "id": "differential_expression",
    "name": "Differential expression (PyDESeq2 or eBayes-OLS)",
    "category": "analysis",
    "description": "Gene-level DE between two levels of a condition column, adjusting for covariates "
                   "(e.g. batch). Outputs a ranked table with log2FC, p, BH-adjusted p.",
    "inputs": {
        "counts": {"type": "path", "required": True},
        "metadata": {"type": "path", "required": True},
        "condition": {"type": "str", "required": True},
        "contrast": {"type": "list", "required": True, "description": "[test_level, reference_level]"},
        "covariates": {"type": "list", "description": "metadata columns added to the design before the condition"},
        "exclude_samples": {"type": "list"},
        "sample_col": {"type": "str", "default": "sample"},
        "method": {"type": "str", "default": "auto"},
        "alpha": {"type": "float", "default": 0.05},
    },
    "outputs": {"results": "{outdir}/de_results.tsv", "summary": "{outdir}/summary.json"},
    "qc_parser": "differential_expression",
}

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def trigamma_inverse(x: float) -> float:
    # Newton iteration used by limma::trigammaInverse
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if -dif / y < 1e-8:
            break
    return float(y)


def squeeze_var(s2: np.ndarray, df: float):
    ok = s2 > 0
    z = np.log(s2[ok])
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean, evar = e.mean(), e.var(ddof=1)
    evar -= special.polygamma(1, df / 2)
    if evar > 0:
        d0 = 2 * trigamma_inverse(evar)
        s0 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s0 = np.inf, np.exp(emean)
    if np.isinf(d0):
        return np.full_like(s2, s0), np.inf, s0
    return (d0 * s0 + df * s2) / (d0 + df), d0, s0


def design_matrix(meta: pd.DataFrame, condition: str, contrast, covariates):
    test, ref = contrast
    cols = {"intercept": np.ones(len(meta))}
    for c in covariates:
        try:
            v = pd.to_numeric(meta[c])
            cols[c] = (v - v.mean()).to_numpy(float)
        except (ValueError, TypeError):
            d = pd.get_dummies(meta[c].astype(str), prefix=c, drop_first=True).astype(float)
            for k in d.columns:
                cols[k] = d[k].to_numpy()
    cols[f"{condition}_{test}_vs_{ref}"] = (meta[condition] == test).to_numpy(float)
    X = pd.DataFrame(cols, index=meta.index)
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise SystemExit(f"design matrix is rank deficient (columns {list(X.columns)}): "
                         f"a covariate is aliased with '{condition}'. Cannot estimate the effect.")
    return X


def run_ebayes(counts, meta, condition, contrast, covariates, alpha):
    lib = counts.sum(axis=0)
    cpm = counts / lib * 1e6
    min_n = int(meta[condition].value_counts().min())
    keep = (cpm > 1).sum(axis=1) >= min_n
    c = counts[keep]
    y = np.log2((c + 0.5) / (lib + 1.0) * 1e6).to_numpy()  # genes x samples
    X = design_matrix(meta, condition, contrast, covariates).to_numpy()
    n, p = X.shape
    df = n - p
    if df < 1:
        raise SystemExit(f"no residual degrees of freedom (n={n}, p={p})")
    XtX_inv = np.linalg.inv(X.T @ X)
    B = y @ X @ XtX_inv  # genes x p
    resid = y - B @ X.T
    s2 = (resid**2).sum(axis=1) / df
    s2_post, d0, s0 = squeeze_var(s2, df)
    j = p - 1
    se = np.sqrt(s2_post * XtX_inv[j, j])
    t = B[:, j] / se
    dft = df + (d0 if np.isfinite(d0) else 1e6)
    pv = 2 * stats.t.sf(np.abs(t), dft)
    res = pd.DataFrame({"gene": c.index, "baseMean": c.mean(axis=1).to_numpy(), "log2FoldChange": B[:, j],
                        "stat": t, "pvalue": pv, "padj": bh(pv)}).set_index("gene")
    info = {"prior_df": None if not np.isfinite(d0) else round(float(d0), 3), "prior_var": round(float(s0), 5),
            "residual_df": df, "genes_tested": int(keep.sum()), "genes_filtered": int((~keep).sum())}
    return res, info


def run_pydeseq2(counts, meta, condition, contrast, covariates, alpha):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    design = "~ " + " + ".join([*covariates, condition])
    dds = DeseqDataSet(counts=counts.T.astype(int), metadata=meta, design=design, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=[condition, *contrast], alpha=alpha, quiet=True)
    st.summary()
    res = st.results_df[["baseMean", "log2FoldChange", "stat", "pvalue", "padj"]].copy()
    res.index.name = "gene"
    return res, {"design": design}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--contrast", nargs=2, required=True)
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--exclude_samples", nargs="*", default=[])
    ap.add_argument("--sample_col", default="sample")
    ap.add_argument("--method", default="auto", choices=["auto", "pydeseq2", "ebayes"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(a.counts, sep="\t", index_col=0)
    meta = pd.read_csv(a.metadata, sep="\t", dtype=str).set_index(a.sample_col)
    samples = [s for s in counts.columns if s not in set(a.exclude_samples)]
    counts, meta = counts[samples], meta.loc[samples]
    for lvl in a.contrast:
        if lvl not in set(meta[a.condition]):
            raise SystemExit(f"contrast level '{lvl}' not in {sorted(set(meta[a.condition]))}")
    meta = meta[meta[a.condition].isin(a.contrast)]
    counts = counts[list(meta.index)]

    method = a.method
    if method == "auto":
        try:
            import pydeseq2  # noqa: F401
            method = "pydeseq2"
        except ImportError:
            method = "ebayes"
    fn = run_pydeseq2 if method == "pydeseq2" else run_ebayes
    res, info = fn(counts, meta, a.condition, a.contrast, a.covariates, a.alpha)
    res = res.sort_values("pvalue")
    res.to_csv(out / "de_results.tsv", sep="\t")

    p = res["pvalue"].dropna().to_numpy()
    pi0 = float(min(1.0, (p > 0.5).mean() / 0.5)) if len(p) else None  # Storey estimate at lambda=0.5
    hist, _ = np.histogram(p, bins=20, range=(0, 1))
    summary = {
        "method": method, "design": "~ " + " + ".join([*a.covariates, a.condition]),
        "contrast": a.contrast, "n_samples": int(counts.shape[1]), "excluded": a.exclude_samples,
        "n_tested": int(res["padj"].notna().sum()),
        "n_significant": int((res["padj"] < a.alpha).sum()),
        "n_up": int(((res["padj"] < a.alpha) & (res["log2FoldChange"] > 0)).sum()),
        "n_down": int(((res["padj"] < a.alpha) & (res["log2FoldChange"] < 0)).sum()),
        "alpha": a.alpha, "pi0": pi0, "pvalue_hist": hist.tolist(),
        "top_genes": res.head(15).reset_index()[["gene", "log2FoldChange", "padj"]].round(5).to_dict("records"),
        **info,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"{method}: {summary['n_significant']} genes padj<{a.alpha} (design {summary['design']}), pi0={pi0}")


if __name__ == "__main__":
    main()
