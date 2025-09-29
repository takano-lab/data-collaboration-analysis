from __future__ import annotations

import argparse
import gzip
import io
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

SERIES_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE29nnn/GSE29272/matrix/GSE29272_series_matrix.txt.gz"


def download_series_matrix(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return dest


def parse_series_matrix(path: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """Return (expr_df, gsm_order, header_fields)
    expr_df: rows=probes, cols=GSM IDs, values=float
    gsm_order: list of GSM IDs in the file order
    header_fields: dict of header arrays like '!Sample_title'
    """
    # Read all text
    raw = path.read_bytes()
    if path.suffix == ".gz":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
            text = f.read().decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    # Collect header arrays
    header_fields: Dict[str, List[str]] = {}
    gsm_order: List[str] = []
    for line in text.splitlines():
        if line.startswith("!Sample_"):
            # Format: !Sample_title = xxx
            if " = " in line:
                key, val = line.split(" = ", 1)
                header_fields.setdefault(key, []).append(val.strip())
        elif line.startswith("!series_matrix_table_begin"):
            break

    # Table part: skip metadata lines starting with '!' or '#'
    df = pd.read_csv(io.StringIO(text), sep='\t', comment='!', header=0)
    # Expect first column to be ID_REF, others GSM ids
    if 'ID_REF' not in df.columns:
        # try alternative first column name
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'ID_REF'})
    df = df.set_index('ID_REF')

    # Keep only numeric columns (GSM IDs)
    # Some columns may be 'Gene title' etc.; GSM IDs start with GSM
    gsm_cols = [c for c in df.columns if str(c).startswith('GSM')]
    if not gsm_cols:
        # if none matched, keep all except non-numeric
        gsm_cols = list(df.columns)
    expr = df[gsm_cols].apply(pd.to_numeric, errors='coerce')

    # Column order
    gsm_order = list(expr.columns)
    return expr, gsm_order, header_fields


def make_labels(gsm_order: List[str], header_fields: Dict[str, List[str]]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Create sample labels using accessions and multiple annotation fields.

    Priority: combine title + source_name + all characteristics, then extract
    keywords (normal / non-cardia / cardia / tumor). Align by accession.
    """
    accessions = [str(a).strip().strip('"') for a in header_fields.get('!Sample_geo_accession', [])]
    N = len(accessions)
    def get_field(name: str) -> List[str]:
        vals = header_fields.get(name, [])
        if len(vals) == N:
            return [str(v).strip().strip('"') for v in vals]
        return [""] * N

    titles = get_field('!Sample_title')
    sources = get_field('!Sample_source_name_ch1')
    # collect characteristics lists with correct length
    char_keys = [k for k, v in header_fields.items() if k.startswith('!Sample_characteristics') and len(v) == N]
    char_vals = {k: [str(x).strip().strip('"') for x in header_fields[k]] for k in char_keys}

    acc_to_label: Dict[str, str] = {}
    for i in range(N):
        parts = [titles[i], sources[i]]
        for k in char_keys:
            parts.append(char_vals[k][i])
        s = " ".join(parts).lower()
        if 'non-cardia' in s or 'noncardia' in s:
            lab = 'non-cardia'
        elif 'cardia' in s:
            lab = 'cardia'
        elif 'normal' in s or 'adjacent tissue normal' in s:
            lab = 'normal'
        elif 'tumor' in s:
            lab = 'tumor'
        else:
            lab = 'unknown'
        acc_to_label[accessions[i]] = lab

    # Build y_names aligned to gsm_order
    y_names: List[str] = [acc_to_label.get(gsm, 'unknown') for gsm in gsm_order]

    # Encode to ints
    classes = sorted(set(y_names))
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_int[c] for c in y_names], dtype=int)
    int_to_cls = {i: c for c, i in cls_to_int.items()}
    return y, int_to_cls


def eval_auc_linear_svc(X: np.ndarray, y: np.ndarray, class_names: Dict[int, str], seed: int = 0) -> Dict[str, float]:
    results: Dict[str, float] = {}

    # Multiclass macro AUC (one-vs-rest)
    k = len(np.unique(y))
    if k >= 2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs: List[float] = []
        for tr, te in skf.split(X, y):
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("svc", SVC(kernel="linear", probability=False, random_state=seed))
            ])
            pipe.fit(X[tr], y[tr])
            # decision_function supports ovR scores
            dec = pipe.decision_function(X[te])
            if k == 2:
                # shape (n_samples,), convert to 2d
                dec = dec.reshape(-1, 1)
                Yte = label_binarize(y[te], classes=np.unique(y))
                auc = roc_auc_score(Yte, dec, average='macro')
            else:
                Yte = label_binarize(y[te], classes=np.unique(y))
                auc = roc_auc_score(Yte, dec, average='macro', multi_class='ovr')
            aucs.append(float(auc))
        results['auc_macro_ovr'] = float(np.mean(aucs))

    # Binary: cardia vs non-cardia (exclude normal and others)
    # Map class names
    inv = {v: k for k, v in class_names.items()}
    if 'cardia' in inv and 'non-cardia' in inv:
        mask = np.isin(y, [inv['cardia'], inv['non-cardia']])
        Xb, yb = X[mask], y[mask]
        yb_bin = (yb == inv['cardia']).astype(int)
        if len(np.unique(yb_bin)) == 2:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            aucs: List[float] = []
            for tr, te in skf.split(Xb, yb_bin):
                pipe = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("svc", SVC(kernel="linear", probability=False, random_state=seed))
                ])
                pipe.fit(Xb[tr], yb_bin[tr])
                dec = pipe.decision_function(Xb[te])  # shape (n,)
                aucs.append(float(roc_auc_score(yb_bin[te], dec)))
            results['auc_cardia_vs_noncardia'] = float(np.mean(aucs))

    # Binary: tumor vs normal (normal vs others)
    if 'normal' in inv:
        normal_id = inv['normal']
        yb_bin2 = (y != normal_id).astype(int)
        if len(np.unique(yb_bin2)) == 2:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            aucs2: List[float] = []
            for tr, te in skf.split(X, yb_bin2):
                pipe = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("svc", SVC(kernel="linear", probability=False, random_state=seed))
                ])
                pipe.fit(X[tr], yb_bin2[tr])
                dec = pipe.decision_function(X[te])
                aucs2.append(float(roc_auc_score(yb_bin2[te], dec)))
            results['auc_tumor_vs_normal'] = float(np.mean(aucs2))

    return results


def main():
    ap = argparse.ArgumentParser(description="Compute Linear SVC AUCs on GSE29272")
    ap.add_argument('--out', type=str, default='output/gse29272_svc_auc')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / 'GSE29272_series_matrix.txt.gz'

    print('Downloading series matrix...')
    fpath = download_series_matrix(SERIES_URL, cache_path)
    print(f'Downloaded to {fpath}')

    print('Parsing series matrix...')
    expr_df, gsm_order, header_fields = parse_series_matrix(fpath)
    print(f'Expression shape: {expr_df.shape}, samples: {len(gsm_order)}')

    y, int_to_cls = make_labels(gsm_order, header_fields)
    print('Label distribution:', {int_to_cls[i]: int((y==i).sum()) for i in sorted(int_to_cls)})

    # Align X by samples (columns)
    X = expr_df.to_numpy(dtype=float).T  # samples x features
    # Filter features with any NaN
    mask_feat = ~np.isnan(X).any(axis=0)
    X = X[:, mask_feat]
    print(f'Features after NaN filter: {X.shape[1]}')

    results = eval_auc_linear_svc(X, y, int_to_cls, seed=args.seed)
    print('AUC results:', results)

    pd.DataFrame([results]).to_csv(out_dir / 'auc_results.csv', index=False)


if __name__ == '__main__':
    main()
