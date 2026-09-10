"""D1 pilot: fixed test splits, fold-local preprocessing, randomized tuning and OOF thresholds."""
import argparse, importlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.preprocessing import (load_raw_dataset, clean_before_split, FEATURE_COLS, fit_bmi_mean, apply_bmi_imputation, fit_iqr_bounds, apply_iqr_clipping, build_preprocessor)

class Preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = fit_bmi_mean(X)
        z = apply_bmi_imputation(X, self.mean_)
        self.bounds_ = fit_iqr_bounds(z)
        self.encoder_ = build_preprocessor().fit(apply_iqr_clipping(z, self.bounds_))
        return self
    def transform(self, X):
        z = apply_iqr_clipping(apply_bmi_imputation(X, self.mean_), self.bounds_)
        return np.asarray(self.encoder_.transform(z), dtype=np.float32)

def metrics(y, prob, threshold):
    pred = prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred),f1=f1_score(y,pred),specificity=float(tn/(tn+fp)),roc_auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),test_rows=len(y))

def choose_threshold(y, prob, specificity):
    fpr,tpr,thresholds=roc_curve(y,prob,drop_intermediate=False)
    valid=np.flatnonzero((1-fpr >= specificity) & np.isfinite(thresholds))
    # Highest recall; tie-break by specificity then larger threshold.
    best=max(valid,key=lambda i:(tpr[i],1-fpr[i],thresholds[i]))
    return float(thresholds[best]),float(tpr[best]),float(1-fpr[best])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--trials',type=int,default=24)
    args=ap.parse_args()
    out=Path('results/dataset1/12_d1_tuning');out.mkdir(parents=True,exist_ok=True)
    df=clean_before_split(load_raw_dataset(args.data,'dataset1'),'dataset1')
    X=df[FEATURE_COLS]; y=df.stroke.to_numpy(dtype='int8')
    cv=StratifiedKFold(3,shuffle=True,random_state=2026)
    space={'model__n_estimators':[200,400,800], 'model__max_depth':[2,3,4,5,6,8],
           'model__learning_rate':[.03,.05,.1], 'model__min_child_weight':[1,3,5],
           'model__subsample':[.8,1.], 'model__colsample_bytree':[.8,1.],
           'model__reg_lambda':[1.,5.,10.], 'model__reg_alpha':[0.,.1,1.]}
    summary=[]; selected=[]
    old=importlib.import_module('experiments.11_resampling_order')
    for protocol in ['split_then_resample','resample_then_split_inferred']:
        if protocol=='split_then_resample':
            tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
            xt,xe,yt,ye=X.iloc[tr],X.iloc[te],y[tr],y[te]
        else:
            encoded,_=old.transform(X,X.iloc[:1],'dataset1')
            xb,yb=SMOTE(random_state=42).fit_resample(encoded,y)
            xt,xe,yt,ye=train_test_split(xb,yb,test_size=.2,random_state=42)
        # Finish model/threshold selection for BOTH PCA options before test evaluation.
        frozen=[]
        for pca in [False,True]:
            steps=[]
            if protocol=='split_then_resample':
                steps += [('preprocess',Preprocess()),('sampler',SMOTE(random_state=42))]
            if pca: steps += [('pca',PCA(n_components=.95,svd_solver='full'))]
            steps += [('model',XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=1,random_state=42))]
            start=time.perf_counter()
            search=RandomizedSearchCV(Pipeline(steps),space,n_iter=args.trials,scoring={'roc_auc':'roc_auc','average_precision':'average_precision'},refit='roc_auc',cv=cv,n_jobs=4,random_state=2026,error_score='raise')
            search.fit(xt,yt)
            pd.DataFrame(search.cv_results_).to_csv(out/f'{protocol}_pca_{int(pca)}_search.csv',index=False)
            best=search.best_estimator_
            oof=cross_val_predict(best,xt,yt,cv=cv,method='predict_proba',n_jobs=4)[:,1]
            thresholds=[dict(policy='default_0.5',threshold=.5)]
            for specificity in [.8,.85,.9]:
                t,r,s=choose_threshold(yt,oof,specificity)
                thresholds.append(dict(policy=f'oof_specificity_{specificity:.2f}',threshold=t,oof_recall=r,oof_specificity=s))
            info=dict(protocol=protocol,pca=pca,best_cv_auc=search.best_score_,params=search.best_params_,thresholds=thresholds,seconds=time.perf_counter()-start)
            selected.append(info);frozen.append((best,info))
            print('SELECTED '+json.dumps(info),flush=True)
        (out/f'{protocol}_frozen_selection.json').write_text(json.dumps(selected[-2:],indent=2))
        for best,info in frozen:
            prob=best.predict_proba(xe)[:,1]
            np.savez_compressed(out/f'{protocol}_pca_{int(info["pca"])}_predictions.npz',y=ye,prob=prob)
            for setting in info['thresholds']:
                row=dict(protocol=protocol,pca=info['pca'],cv_auc=info['best_cv_auc'],**setting,**metrics(ye,prob,setting['threshold']))
                summary.append(row)
                print('TEST '+json.dumps(row),flush=True)
        pd.DataFrame(summary).to_csv(out/'test_comparison.csv',index=False)
    import sklearn,xgboost,imblearn,platform
    metadata=dict(seed_split=42,seed_search=2026,trials_per_arm_pca=args.trials,cv_folds=3,selection_metric='roc_auc',threshold_selection='Training OOF only, maximize recall subject to OOF specificity target; target not guaranteed on test',caveats=['Prior test already observed; exploratory comparison, not fresh external validation','OOF uses hyperparameters selected on same training CV, not nested CV; OOF optimism possible','Inferred paper arm has full-data preprocessing and pre-split SMOTE; not unbiased validation','No SMOTENC, class weights, IQR alternatives or PCA 99% in this first tuning stage'],versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,xgboost=xgboost.__version__,imblearn=imblearn.__version__))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
