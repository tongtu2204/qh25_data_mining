"""Compare D1 SMOTE/SMOTENC/weights and D2 undersampling/full-train weights.

Run from repo root. Split original data before fitting or resampling. Models
and threshold choices are locked before evaluating the unchanged original test.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from src.preprocessing import load_raw_dataset, clean_before_split, FEATURE_COLS
from src.improvements import build_pipeline, parameter_space, select_threshold, evaluate


def save_json(path, value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,choices=['dataset1','dataset2'])
    ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--trials',type=int,default=12)
    ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--tuning-rows',type=int,default=150000)
    args=ap.parse_args()
    if args.trials<1 or args.jobs<1: ap.error('trials and jobs must be positive')
    root=Path(__file__).resolve().parents[1]
    out=root/'results'/args.dataset/'13_balancing_improvements'
    out.mkdir(parents=True,exist_ok=True)
    # Models/predictions are local ignored artifacts, not committed raw records.
    local=out/'local';local.mkdir(exist_ok=True)
    digest=hashlib.sha256()
    with args.data.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
    raw=load_raw_dataset(args.data,args.dataset)
    missing=raw[['stroke','hypertension','heart_disease']].isna().sum().to_dict()
    if any(missing.values()): raise ValueError(f'Binary missing values need an explicit policy: {missing}')
    raw_n=len(raw)
    df=clean_before_split(raw,args.dataset);del raw
    if df.stroke.nunique()!=2: raise ValueError('Both target classes required')
    X=df[FEATURE_COLS];y=df.stroke.to_numpy(dtype='int8')
    itr,ite=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
    xt,xe,yt,ye=X.iloc[itr],X.iloc[ite],y[itr],y[ite]
    np.savez_compressed(local/'split.npz',train_indices=itr,test_indices=ite,y_test=ye)
    cv=StratifiedKFold(3,shuffle=True,random_state=2026)
    if args.dataset=='dataset2':
        fit_idx,val_idx=train_test_split(np.arange(len(yt)),test_size=.2,random_state=2026,stratify=yt)
        xs,ys=xt.iloc[fit_idx],yt[fit_idx]
        if len(ys)>args.tuning_rows:
            sub,_=train_test_split(np.arange(len(ys)),train_size=args.tuning_rows,random_state=2026,stratify=ys)
            xs,ys=xs.iloc[sub],ys[sub]
        validation_source='heldout 20% of outer training; excluded from tuning'
    else:
        xs,ys=xt,yt
        validation_source='3-fold training OOF after hyperparameter selection; not nested CV'
    metadata=dict(run_status='running',dataset=args.dataset,raw_rows=raw_n,cleaned_rows=len(y),train_rows=len(yt),test_rows=len(ye),
        train_positive=int(yt.sum()),test_positive=int(ye.sum()),input_sha256=digest.hexdigest(),
        input_size_bytes=args.data.stat().st_size,
        input_source=('project-supplied healthcare-dataset-stroke-data.csv' if args.dataset=='dataset1' else
                      'Kaggle pranavp1999/stroke-prediction-health-care-synthetic-dataset ZIP'),
        seed_split=42,seed_cv_search=2026,trials_per_variant=args.trials,cv_folds=3,
        search_rows=len(ys),selection_metric='standardized partial ROC-AUC at max FPR=0.10',
        threshold_source=validation_source,specificity_targets=[.85,.9,.95],
        caveats=['Same previously observed test split; exploratory comparison, not external validation',
        'D1 OOF after selection is not nested and may be optimistic',
        'Specificity targets apply to threshold-selection data, not guaranteed on test',
        'D2 thresholds transferred from a model fit on inner train to full-train refit',
        'D2 binary threshold 0.5 and basic cleaning retained from repo for comparability'],
        missing_binary=missing)
    import sklearn,xgboost,imblearn
    metadata['versions']=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,imblearn=imblearn.__version__)
    save_json(out/'metadata.json',metadata)
    print('DATA '+json.dumps(metadata),flush=True)
    methods=['smote','smotenc','weighted'] if args.dataset=='dataset1' else ['undersampling','weighted']
    selected=[];val_rows=[]
    # Phase 1: all selection/final fitting before reading any test predictions.
    for method in methods:
        for pca in [False,True]:
            name=f'{method}_pca_{int(pca)}';start=time.perf_counter()
            pipe=build_pipeline(method,pca)
            search=RandomizedSearchCV(pipe,parameter_space(method,pca),n_iter=args.trials,
                scoring={'partial_auc_90':make_scorer(roc_auc_score,response_method='predict_proba',max_fpr=.1),
                         'roc_auc':'roc_auc','average_precision':'average_precision'},
                refit='partial_auc_90',cv=cv,n_jobs=args.jobs,random_state=2026,error_score='raise')
            search.fit(xs,ys)
            pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_search.csv',index=False)
            best=search.best_estimator_
            if args.dataset=='dataset1':
                pv=cross_val_predict(best,xt,yt,cv=cv,method='predict_proba',n_jobs=args.jobs)[:,1]
                yv=yt
                final=best  # search refitted on full outer train
            else:
                inner=clone(best).set_params(model__n_jobs=args.jobs)
                inner.fit(xt.iloc[fit_idx],yt[fit_idx])
                pv=inner.predict_proba(xt.iloc[val_idx])[:,1];yv=yt[val_idx]
                del inner
                final=clone(best).set_params(model__n_jobs=args.jobs)
                final.fit(xt,yt)
            policies=[dict(policy='default_0.5',threshold=.5)]
            for s in [.85,.9,.95]:
                policies.append(dict(policy=f'specificity_{s:.2f}',threshold=select_threshold(yv,pv,s)))
            info=dict(name=name,method=method,pca=pca,cv_partial_auc_90=search.best_score_,
                params=search.best_params_,policies=policies,seconds=time.perf_counter()-start,
                final_train_rows=len(yt),uses_no_resampling=(method=='weighted'),retains_all_original_training_rows=(method!='undersampling'))
            for policy in policies:
                val_rows.append(dict(name=name,method=method,pca=pca,**policy,**evaluate(yv,pv,policy['threshold'])))
            joblib.dump(final,local/f'{name}.joblib')
            np.savez_compressed(local/f'{name}_validation.npz',y=yv,prob=pv)
            selected.append(info)
            save_json(out/'selected_models.json',selected)
            pd.DataFrame(val_rows).to_csv(out/'validation_comparison.csv',index=False)
            print('SELECTED '+json.dumps(info),flush=True)
            del final,best,search
    winners={}
    for s in [.85,.9,.95]:
        policy=f'specificity_{s:.2f}'
        candidates=[r for r in val_rows if r['policy']==policy and r['specificity']>=s-1e-12]
        win=max(candidates,key=lambda r:(r['recall'],r['specificity'],r['average_precision']))
        winners[policy]=dict(name=win['name'],threshold=win['threshold'],validation_recall=win['recall'],validation_specificity=win['specificity'])
    save_json(out/'frozen_winners.json',winners)
    # Phase 2: evaluate all prespecified variants/policies, never choose on test.
    test_rows=[]
    for info in selected:
        model=joblib.load(local/f"{info['name']}.joblib")
        prob=model.predict_proba(xe)[:,1]
        np.savez_compressed(local/f"{info['name']}_test.npz",y=ye,prob=prob)
        for policy in info['policies']:
            row=dict(name=info['name'],method=info['method'],pca=info['pca'],**policy,
                selected_on_validation=winners.get(policy['policy'],{}).get('name')==info['name'],
                **evaluate(ye,prob,policy['threshold']))
            test_rows.append(row);print('TEST '+json.dumps(row),flush=True)
        pd.DataFrame(test_rows).to_csv(out/'test_comparison.csv',index=False)
        del model
    metadata['run_status']='completed'
    metadata['test_result_rows']=len(test_rows)
    save_json(out/'metadata.json',metadata)
    print('COMPLETE '+args.dataset,flush=True)

if __name__=='__main__': main()
