from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
SEED = 42

DROP = ['company', 'industry', 'link', 'num', 'emis_id', 'sector_2', 'sector_3', 'sector_4',
        'Revenue/employee', 'Fixed_assets/employee', 'EBITDA/cash_flow']
NOT_FEATURES = ['operational_status', 'year', 'main_label']


def load_horizon(h, keep=()):
    df = pd.read_parquet(DATA / f'company_years_h{h + 1}.parquet')
    y = df.main_label.astype(int).to_numpy()
    groups = df.company.to_numpy()
    remove = DROP + [c for c in NOT_FEATURES if c not in keep]
    X = df.drop(columns=remove).select_dtypes('number').astype('float32')
    return X, y, groups

def make_folds(y, groups, n=5, seed=SEED):
    fold = np.full(len(y), -1)
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold[test_idx] = k
    assert pd.Series(fold).groupby(groups).nunique().max() == 1, 'company in two folds'
    return fold


def best_threshold(y_true, score):
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2 * p * r / (p + r + 1e-12)
    return t[np.argmax(f1[:-1])]


def evaluate(y_true, score, thr):
    pred = (score >= thr).astype(int)
    return dict(roc_auc=roc_auc_score(y_true, score),
                pr_auc=average_precision_score(y_true, score),
                pr_baseline=y_true.mean(),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred),
                f1=f1_score(y_true, pred),
                threshold=thr)


def run_cv(name, h, X, y, fold, make_model=None, score_fn=None, verbose=True):
    """Fold k = test, fold k+1 = validation (threshold), the other 3 = training."""
    rows = []
    for k in range(5):
        te, va = fold == k, fold == (k + 1) % 5
        tr = ~(te | va)
        if make_model is not None:
            m = make_model().fit(X[tr], y[tr])
            s_va, s_te = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
        else:
            s_va, s_te = score_fn(X[tr], X[va], X[te])
        thr = best_threshold(y[va], s_va)
        rows.append({'model': name, 'horizon': h, 'fold': k, **evaluate(y[te], s_te, thr)})
        if verbose:
            print(f'{name} h={h} fold {k}: ROC-AUC {rows[-1]["roc_auc"]:.3f}')
    return pd.DataFrame(rows)