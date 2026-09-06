"""Pilot resampling audit. Paper order is INFERRED, not verified author code.

Run from repo root with --dataset dataset1|dataset2 --data PATH.
No test-based tuning: reuse the repo's saved tuned XGBoost parameters.
The pre-split SMOTE arm is deliberately diagnostic and is not an unbiased test.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, average_precision_score,
                            matthews_corrcoef, confusion_matrix)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.preprocessing import (load_raw_dataset, clean_before_split,
    FEATURE_COLS, fit_bmi_mean, apply_bmi_imputation, fit_iqr_bounds,
    apply_iqr_clipping, fit_transform_features)


def transform(train, test, dataset):
    if dataset == 'dataset1':
        mean = fit_bmi_mean(train)
        train = apply_bmi_imputation(train, mean)
        test = apply_bmi_imputation(test, mean)
    bounds = fit_iqr_bounds(train)
    train = apply_iqr_clipping(train, bounds)
    test = apply_iqr_clipping(test, bounds)
    a, b, _, _ = fit_transform_features(train, test)
    return a, b


def evaluate(y, prob):
    pred = (prob >= .5).astype('int8')
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(test_rows=len(y), test_positive=int(np.sum(y)),
        prevalence=float(np.mean(y)), accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, prob),
        average_precision=average_precision_score(y, prob),
        mcc=matthews_corrcoef(y, pred), specificity=float(tn/(tn+fp)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['dataset1', 'dataset2'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root/'results'/args.dataset/'11_resampling_order'/f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw_dataset(args.data, args.dataset)
    missing_binary = raw[['stroke', 'hypertension', 'heart_disease']].isna().sum().to_dict()
    # Preserve the existing cleaning/label policy to isolate resampling order.
    # If binary missing values exist, fail instead of silently converting them to 0.
    if any(missing_binary.values()):
        raise ValueError(f'Binary missing values require a separate cleaning decision: {missing_binary}')
    df = clean_before_split(raw, args.dataset)
    raw_rows = len(raw)
    del raw
    y = df.stroke.to_numpy(dtype='int8')
    X = df[FEATURE_COLS]
    params = json.loads((root/'results'/args.dataset/'05_hyperparameter_search'/'best_params.json').read_text())['tuning']['best_params']
    metadata = dict(dataset=args.dataset, seed=args.seed, raw_rows=raw_rows,
        cleaned_rows=len(y), cleaned_positive=int(y.sum()), missing_binary=missing_binary,
        parameters=params, tuning='No new tuning; reuse prior saved parameters',
        threshold=.5, paper_order='INFERRED; not confirmed by author source code',
        assumptions=['Existing repo encoding, IQR clipping and D2 binary threshold 0.5 retained',
          'Paper seed, exact encoding, sampling subset and SMOTE neighbors unspecified; seed 42, k=5 assumed',
          'Unstratified 80/20 split in inferred paper arm; stratified 80/20 in control',
          'PCA fitted after resampling on training data only in both arms',
          'No new CV or grid search in this pilot; no independent final performance claim',
          'Pre-split preprocessing/SMOTE arm may leak information across its split'],
        versions=dict(python=platform.python_version(), numpy=np.__version__,
          pandas=pd.__version__, sklearn=sklearn.__version__,
          xgboost=xgboost.__version__, imblearn=imblearn.__version__))
    rows = []
    for protocol in ['split_then_resample', 'resample_then_split_inferred']:
        sampler = (SMOTE(random_state=args.seed, k_neighbors=5) if args.dataset == 'dataset1'
                   else RandomUnderSampler(random_state=args.seed, sampling_strategy=1.0))
        if protocol == 'split_then_resample':
            itr, ite = train_test_split(np.arange(len(y)), test_size=.2, random_state=args.seed, stratify=y)
            train, test = transform(X.iloc[itr], X.iloc[ite], args.dataset)
            yt, ye = y[itr], y[ite]
            train, yt = sampler.fit_resample(train, yt)
            test_kind = 'original_test'
        else:
            if args.dataset == 'dataset1':
                encoded, _ = transform(X, X.iloc[:1], args.dataset)
                balanced, balanced_y = sampler.fit_resample(encoded, y)
                del encoded
                train, test, yt, ye = train_test_split(balanced, balanced_y,
                    test_size=.2, random_state=args.seed)
            else:
                balanced, balanced_y = sampler.fit_resample(X, y)
                atr, ate, yt, ye = train_test_split(balanced, balanced_y,
                    test_size=.2, random_state=args.seed)
                train, test = transform(atr, ate, args.dataset)
            del balanced, balanced_y
            test_kind = 'resampled_test'
        np.savez_compressed(out/f'{protocol}_labels.npz', y_test=ye)
        for use_pca in [False, True]:
            start = time.perf_counter()
            if use_pca:
                pca = PCA(n_components=.95, svd_solver='full')
                a = pca.fit_transform(train).astype('float32')
                b = pca.transform(test).astype('float32')
            else:
                a, b = train, test
            model = XGBClassifier(**params, objective='binary:logistic',
                eval_metric='logloss', tree_method='hist', n_jobs=args.jobs, random_state=args.seed)
            model.fit(a, yt)
            prob = model.predict_proba(b)[:, 1]
            elapsed = time.perf_counter()-start
            base = dict(dataset=args.dataset, protocol=protocol, pca=use_pca,
                train_rows=len(yt), train_positive=int(np.sum(yt)), features=a.shape[1],
                elapsed_seconds=elapsed)
            row = dict(**base, evaluation=test_kind, **evaluate(ye, prob))
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.save_model(out/f'{protocol}_pca_{int(use_pca)}.json')
            np.savez_compressed(out/f'{protocol}_pca_{int(use_pca)}_predictions.npz', y=ye, probability=prob)
            if protocol == 'split_then_resample':
                indices, y_bal = RandomUnderSampler(random_state=args.seed).fit_resample(np.arange(len(ye)).reshape(-1, 1), ye)
                row = dict(**base, evaluation='balanced_real_test_same_model',
                           **evaluate(y_bal, prob[indices.ravel()]))
                rows.append(row)
                print(json.dumps(row), flush=True)
            pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
            del model, a, b
        del train, test
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
