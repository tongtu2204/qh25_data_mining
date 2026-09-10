"""Fold-local preprocessing and evaluation helpers for experiment 13."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    matthews_corrcoef)
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from src.preprocessing import NUMERIC_COLS, BINARY_FEATURE_COLS, CATEGORICAL_COLS

class MixedPreprocessor(BaseEstimator, TransformerMixin):
    """Numeric scaling + categorical codes, fitted exclusively on training rows.

    Output: three numeric, two binary, five categorical columns. Binary columns
    are treated as categorical by SMOTENC but are not one-hot expanded later.
    """
    def __init__(self, clip_iqr=True):
        self.clip_iqr = clip_iqr
    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy='mean').fit(X[NUMERIC_COLS])
        num = self.imputer_.transform(X[NUMERIC_COLS])
        q1, q3 = np.quantile(num, [.25, .75], axis=0)
        self.lower_, self.upper_ = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        self.scaler_ = StandardScaler().fit(self._clip(num))
        self.ordinal_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
            dtype=np.float32).fit(X[CATEGORICAL_COLS])
        return self
    def _clip(self, num):
        return np.clip(num, self.lower_, self.upper_) if self.clip_iqr else num
    def transform(self, X):
        num = self.scaler_.transform(self._clip(self.imputer_.transform(X[NUMERIC_COLS])))
        return np.column_stack([num, X[BINARY_FEATURE_COLS].to_numpy(),
            self.ordinal_.transform(X[CATEGORICAL_COLS])]).astype(np.float32)


def build_pipeline(method, use_pca, seed=42, jobs=1):
    encode = ColumnTransformer([
        ('numeric_binary', 'passthrough', list(range(5))),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
            dtype=np.float32), list(range(5,10)))])
    steps = [('preprocess', MixedPreprocessor())]
    if method == 'smotenc':
        steps += [('sampler', SMOTENC(categorical_features=list(range(3,10)), random_state=seed)),
                  ('encode', encode)]
    else:
        steps += [('encode', encode)]
        if method == 'smote': steps += [('sampler', SMOTE(random_state=seed))]
        elif method == 'undersampling': steps += [('sampler', RandomUnderSampler(random_state=seed))]
        elif method != 'weighted': raise ValueError(method)
    if use_pca: steps += [('pca', PCA(n_components=.95, svd_solver='full'))]
    steps += [('model', XGBClassifier(objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', random_state=seed, n_jobs=jobs))]
    return Pipeline(steps)


def parameter_space(method, use_pca):
    space = {
        'preprocess__clip_iqr': [True, False],
        'model__n_estimators': [200,400,600], 'model__max_depth': [2,3,5,7],
        'model__learning_rate': [.03,.1], 'model__min_child_weight': [1,5],
        'model__subsample': [.8,1.], 'model__colsample_bytree': [.8,1.],
        'model__reg_lambda': [1.,5.,10.], 'model__reg_alpha': [0.,.5],
    }
    if use_pca: space['pca__n_components'] = [.95, .99, 13]
    if method == 'weighted': space['model__scale_pos_weight'] = [1.,5.,10.,20.]
    else:
        space['sampler__sampling_strategy'] = [.25,.5,1.]
        if method in ['smote','smotenc']: space['sampler__k_neighbors'] = [3,5,7]
    return space


def select_threshold(y, prob, specificity):
    fpr, tpr, thresholds = roc_curve(y, prob, drop_intermediate=False)
    feasible = np.flatnonzero(1-fpr >= specificity)
    i = max(feasible, key=lambda i:(tpr[i], 1-fpr[i], thresholds[i]))
    # Keep an all-negative option even when all finite thresholds violate the target.
    t = thresholds[i]
    if not np.isfinite(t): t = 1.000001
    return float(t)


def evaluate(y, prob, threshold):
    pred = prob >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),positive=int(np.sum(y)),prevalence=float(np.mean(y)),
        accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred),specificity=float(tn/(tn+fp)),f1=f1_score(y,pred),
        roc_auc=roc_auc_score(y,prob),partial_auc_90=roc_auc_score(y,prob,max_fpr=.1),
        average_precision=average_precision_score(y,prob),mcc=matthews_corrcoef(y,pred),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
