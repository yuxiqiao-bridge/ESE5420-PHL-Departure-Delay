#!/usr/bin/env python3
"""ESE 5420 PHL baseline checkpoint, 2025 BTS original monthly on-time data.

Run in JupyterLab terminal:
    pip install pandas numpy scikit-learn matplotlib requests
    python PHL_checkpoint_auto.py

Or provide one/more CSVs locally:
    python PHL_checkpoint_auto.py --data /path/to/csv-or-folder

Regularization C is tuned by five expanding-window, whole-day chronological folds within Jan--Aug.
September--October is used for validation scoring and threshold choice.
Nov--Dec are counted but never used for training, tuning, or scoring.
"""
from __future__ import annotations
import argparse
import re
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE='https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{}.zip'
NEEDED=('YEAR','MONTH','DAYOFMONTH','DAYOFWEEK','ORIGIN','DEST','CRSDEPTIME','CANCELLED','DEPDEL15')
RAW_DIR=Path('BTS_2025_PHL_raw')

def norm(s): return re.sub('[^A-Z0-9]','',str(s).strip().upper())

def load_csv(src, target_dir):
    """Read just needed columns chunkwise; PHL filter before retaining."""
    selected=[]
    for chunk in pd.read_csv(src, chunksize=120_000, low_memory=False, encoding='utf-8-sig'):
        raw={norm(k):k for k in chunk.columns}
        if 'REPORTINGAIRLINE' in raw:
            carrier='REPORTINGAIRLINE'
        elif 'OPUNIQUECARRIER' in raw:
            carrier='OPUNIQUECARRIER'
        elif 'UNIQUECARRIER' in raw:
            carrier='UNIQUECARRIER'
        else: raise ValueError('Missing reporting airline column')
        missing=set(NEEDED)-set(raw)
        if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
        c=chunk.loc[chunk[raw['ORIGIN']].astype(str).str.upper().str.strip()=='PHL',
                    [raw[x] for x in NEEDED]+[raw[carrier]]].copy()
        if len(c):
            c.columns=list(NEEDED)+['AIRLINE']
            selected.append(c)
    if not selected: raise ValueError('No PHL origin flights in '+str(src))
    df=pd.concat(selected,ignore_index=True)
    del selected
    return df

def download_zip(month, raw_dir, timeout=120):
    import requests
    url=BASE.format(month)
    target=raw_dir/f'BTS_2025_{month:02}.zip'
    if target.exists() and zipfile.is_zipfile(target):return target
    for attempt in range(3):
        try:
            print(f'Downloading 2025 month {month:02} (attempt {attempt+1}/3)...',flush=True)
            with requests.get(url,stream=True,timeout=(20,timeout),headers={'User-Agent':'Mozilla/5.0'}) as r:
                r.raise_for_status()
                with open(str(target)+'.part','wb') as w:
                    for block in r.iter_content(1024*1024):
                        if block:w.write(block)
            Path(str(target)+'.part').replace(target)
            if not zipfile.is_zipfile(target):raise ValueError('Downloaded content is not a ZIP archive')
            return target
        except Exception as e:
            print(f'  Download error: {e}',flush=True)
            Path(str(target)+'.part').unlink(missing_ok=True)
            if attempt<2:time.sleep(2*(attempt+1))
    raise RuntimeError('Cannot download '+url+'; download manually from official site and pass --data.')

def acquire(args):
    if args.data:
        path=Path(args.data)
        if path.is_dir(): files=sorted(path.glob('*.csv'))+sorted(path.glob('*.zip'))
        else:files=[path]
        if not files:raise ValueError('No .csv/.zip files found')
        data=[]
        for file in files:
            print('Reading',file,flush=True)
            if file.suffix.lower()=='.zip':
                with zipfile.ZipFile(file) as z:
                    members=[x for x in z.namelist() if x.lower().endswith('.csv')]
                    if not members: raise ValueError(f'No CSV inside {file}')
                    with z.open(members[0]) as f:data.append(load_csv(f,None))
            else:data.append(load_csv(file,None))
        return pd.concat(data,ignore_index=True)
    RAW_DIR.mkdir(exist_ok=True)
    months=[]
    for month in range(1,13):
        cache=RAW_DIR/f'PHL_2025_{month:02}.csv'
        if cache.exists():
            print('Using cached',cache,flush=True)
            months.append(pd.read_csv(cache,low_memory=False))
            continue
        archive=download_zip(month,RAW_DIR)
        print('Reading and filtering',archive.name,flush=True)
        with zipfile.ZipFile(archive) as z:
            members=[x for x in z.namelist() if x.lower().endswith('.csv')]
            if not members:raise ValueError('CSV missing inside '+archive.name)
            with z.open(members[0]) as f:small=load_csv(f,None)
        small.to_csv(cache,index=False)
        months.append(small)
        if not args.keep_zips:archive.unlink(missing_ok=True)
    return pd.concat(months,ignore_index=True)

def cluster_bootstrap(data, columns, reps=1000, seed=42):
    """Sample whole validation days and compute AP (and paired M4-M1 AP)."""
    dates=pd.to_datetime(data['DATE']).dt.strftime('%Y-%m-%d').to_numpy()
    days, inv=np.unique(dates,return_inverse=True)
    groups=[np.flatnonzero(inv==d) for d in range(len(days))]
    yy=data['Y'].to_numpy(dtype=int)
    rng=np.random.default_rng(seed)
    vals={name:[] for name in columns}
    diffs=[]
    for _ in range(reps):
        draw=rng.integers(0,len(days),size=len(days))
        ii=np.concatenate([groups[d] for d in draw])
        if len(np.unique(yy[ii]))<2:continue
        ap={name:average_precision_score(yy[ii],p[ii]) for name,p in columns.items()}
        for name,v in ap.items():vals[name].append(v)
        if 'M1' in ap and 'M4' in ap:diffs.append(ap['M4']-ap['M1'])
    return ({k:np.quantile(v,[0.025,0.975]) for k,v in vals.items()},
            np.quantile(diffs,[0.025,0.975]) if diffs else None, len(vals['M1']))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',help='optional existing .csv/.zip or directory of them')
    parser.add_argument('--out',default='PHL_checkpoint_submission')
    parser.add_argument('--keep-zips',action='store_true')
    parser.add_argument('--team',default='[Canvas group number]; Jiayi, Xinyu, Yuxi')
    parser.add_argument('--code-url',default='[ADD SHARED NOTEBOOK OR GITHUB LINK]')
    parser.add_argument('--bootstrap',type=int,default=400)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--cost-ratio',type=float,default=5.)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    df=acquire(args)
    for k in ['YEAR','MONTH','DAYOFMONTH','DAYOFWEEK','CRSDEPTIME','CANCELLED','DEPDEL15']:
        df[k]=pd.to_numeric(df[k],errors='coerce')
    df=df.loc[(df['YEAR']==2025)&(df['ORIGIN'].astype(str).str.strip().str.upper()=='PHL')].copy()
    raw=len(df)
    dropped_cancelled=int((df['CANCELLED']==1).sum())
    df=df.loc[df['CANCELLED']==0].copy()
    noncancel=len(df)
    df=df.dropna(subset=['YEAR','MONTH','DAYOFMONTH','DAYOFWEEK','CRSDEPTIME','DEPDEL15','DEST','AIRLINE'])
    df['DATE']=pd.to_datetime(dict(year=df.YEAR,month=df.MONTH,day=df.DAYOFMONTH),errors='coerce')
    df=df.dropna(subset=['DATE']).copy()
    df['SCHED_MIN']=(df.CRSDEPTIME//100)*60+(df.CRSDEPTIME%100)
    df=df.loc[df.SCHED_MIN.between(0,1439)&df.DEPDEL15.isin([0,1])].copy()
    df['Y']=df.DEPDEL15.astype(int)
    df['DOW']=df.DAYOFWEEK.astype(int).astype(str)
    df['AIRLINE']=df.AIRLINE.astype(str).str.strip()
    df['DESTINATION']=df.DEST.astype(str).str.strip()
    usable=len(df);dropped_missing=noncancel-usable
    monthly=df.groupby('MONTH')['Y'].agg(N='size',delay_rate='mean').reindex(range(1,13))
    if monthly.N.isna().any():raise ValueError('Not all twelve months found: '+str(monthly.N.to_dict()))
    monthly.to_csv(out/'monthly_summary.csv')
    train=df[df.MONTH<=8].copy()
    val=df[df.MONTH.isin([9,10])].copy()
    test=df[df.MONTH.isin([11,12])].copy()
    assert len(train)>0 and len(val)>0 and len(test)>0
    # Create test only for counts; neither model.fit nor validation scoring accesses test.
    cols_map={'M1':['SCHED_MIN','DOW'],
              'M2':['SCHED_MIN','DOW','AIRLINE'],
              'M3':['SCHED_MIN','DOW','DESTINATION'],
              'M4':['SCHED_MIN','DOW','AIRLINE','DESTINATION']}
    # Split entire calendar days rather than individual flights to avoid leakage.
    train=train.sort_values(['DATE','SCHED_MIN']).reset_index(drop=True)
    unique_days=np.array(sorted(train.DATE.unique()))
    splitter=TimeSeriesSplit(n_splits=args.folds)
    folds=[]
    for fold_id,(td,vd) in enumerate(splitter.split(unique_days),start=1):
        trdays=unique_days[td]; vdays=unique_days[vd]
        itrain=np.flatnonzero(train.DATE.isin(trdays).to_numpy())
        ival=np.flatnonzero(train.DATE.isin(vdays).to_numpy())
        assert train.DATE.iloc[itrain].max()<train.DATE.iloc[ival].min()
        folds.append((fold_id,itrain,ival))
    fold_details=[]
    for fold_id,itrain,ival in folds:
        fold_details.append({'fold':fold_id,
            'train_start':str(train.DATE.iloc[itrain].min().date()),
            'train_end':str(train.DATE.iloc[itrain].max().date()),
            'validation_start':str(train.DATE.iloc[ival].min().date()),
            'validation_end':str(train.DATE.iloc[ival].max().date()),
            'train_N':len(itrain),'validation_N':len(ival)})
    pd.DataFrame(fold_details).to_csv(out/'cv_fold_dates.csv',index=False)
    grid=np.logspace(-3,2,6)
    all_grid=[];pred={};best={}
    def build_model(cols,C):
        categorical=[c for c in cols if c!='SCHED_MIN']
        prep=ColumnTransformer([
            ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),
                                 ('scale',StandardScaler())]),['SCHED_MIN']),
            ('categorical',OneHotEncoder(handle_unknown='ignore'),categorical)])
        return Pipeline([('preprocess',prep),
            ('logistic',LogisticRegression(C=float(C),max_iter=1200,random_state=42))])
    for group,cols in cols_map.items():
        for C in grid:
            fold_aps=[]
            for fold_id,itrain,ival in folds:
                mdl=build_model(cols,C)
                mdl.fit(train.iloc[itrain][cols],train.Y.iloc[itrain])
                pr=mdl.predict_proba(train.iloc[ival][cols])[:,1]
                ap=average_precision_score(train.Y.iloc[ival],pr)
                fold_aps.append(float(ap))
                all_grid.append({'feature_set':group,'C':float(C),'fold':fold_id,
                                 'fold_AP':float(ap)})
            mean_ap=float(np.mean(fold_aps))
            if group not in best or mean_ap>best[group]['CV_AP']:
                best[group]={'C':float(C),'CV_AP':mean_ap,
                    'CV_AP_SD':float(np.std(fold_aps,ddof=1))}
        final=build_model(cols,best[group]['C'])
        final.fit(train[cols],train.Y)
        pred[group]=final.predict_proba(val[cols])[:,1]
        best[group]['AP']=float(average_precision_score(val.Y,pred[group]))
        print(f"{group}: C={best[group]['C']:g}, CV AP={best[group]['CV_AP']:.4f}, validation AP={best[group]['AP']:.4f}",flush=True)
    pd.DataFrame(all_grid).to_csv(out/'C_validation_grid.csv',index=False)
    pd.DataFrame([{'feature_set':g,'best_C':best[g]['C'],
        'CV_mean_AP':best[g]['CV_AP'],'CV_SD_across_folds':best[g]['CV_AP_SD']}
        for g in cols_map]).to_csv(out/'CV_summary.csv',index=False)
    ci,delta_ci,completed=cluster_bootstrap(val,pred,args.bootstrap)
    summary=pd.DataFrame([{'feature_set':g,'best_C':best[g]['C'],
                           'validation_AP':best[g]['AP'],
                           'day_bootstrap_CI_low':ci[g][0],
                           'day_bootstrap_CI_high':ci[g][1]} for g in cols_map])
    summary.to_csv(out/'validation_results.csv',index=False)
    gain=best['M4']['AP']-best['M1']['AP']
    y=val.Y.to_numpy(dtype=int)
    # Threshold is chosen on Sep--Oct only; AP is threshold-free.
    thresholds=np.linspace(0.01,0.99,99)
    costs=[args.cost_ratio*np.sum((y==1)&(pred['M1']<t))+
           np.sum((y==0)&(pred['M1']>=t)) for t in thresholds]
    threshold=float(thresholds[int(np.argmin(costs))])
    yyhat=(pred['M1']>=threshold).astype(int)
    cm=confusion_matrix(y,yyhat,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    # Show whole-day bootstrap 95% confidence intervals on the baseline figure.
    fig,ax=plt.subplots(figsize=(8.3,5.4))
    names=list(cols_map)
    xpos=np.arange(len(names))
    means=np.array([best[g]['AP'] for g in names])
    bounds=np.array([[best[g]['AP']-ci[g][0] for g in names],
                     [ci[g][1]-best[g]['AP'] for g in names]])
    ax.bar(xpos,means,color=['#516b8b','#7895ad','#9ab2c7','#3f806f'],width=.62)
    ax.errorbar(xpos,means,yerr=bounds,fmt='none',ecolor='#222',capsize=6,lw=1.5)
    ax.set_xticks(xpos,names)
    ax.set(ylabel='Average precision (AP)',ylim=(0,.41),
           title='PHL departure delays: Sep-Oct 2025 validation\\n5-fold chronological CV for C; day-bootstrap 95% intervals')
    for i,g in enumerate(names):ax.text(i,means[i]+.047,f'{means[i]:.3f}',ha='center',fontsize=10)
    ax.text(.02,.03,f'Paired M4 - M1: {gain:+.4f} AP (95% CI: {delta_ci[0]:+.4f} to {delta_ci[1]:+.4f})\\nNovember-December held out; not scored',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(out/'baseline.png',dpi=200);plt.close(fig)
    # Optional small chart foregrounding AP increment, validation only.
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    names=list(cols_map)
    means=[best[g]['AP'] for g in names]
    errors=np.array([[best[g]['AP']-ci[g][0] for g in names],
                     [ci[g][1]-best[g]['AP'] for g in names]])
    ax.bar(names,means,yerr=errors,capsize=5,color='#527aa3')
    ax.set(ylabel='Average Precision on Sep–Oct validation',title='PHL: nested feature sets (day-bootstrap 95% CIs)')
    ax.set_ylim(0,min(1.,max(means)*1.4))
    fig.tight_layout();fig.savefig(out/'feature_comparison.png',dpi=200);plt.close(fig)
    monthly_str='; '.join(f'{m}: {int(row.N)} ({row.delay_rate:.1%})' for m,row in monthly.iterrows())
    with open(out/'results_for_report.txt','w',encoding='utf-8') as f:
        f.write(f'2025 PHL original rows: {raw}; cancelled: {dropped_cancelled}; additional removed: {dropped_missing}; usable: {usable}\n')
        f.write(f'Train: {len(train)} Jan-Aug; validation: {len(val)} Sep-Oct; held-out: {len(test)} Nov-Dec\n')
        f.write(f'K-fold: {args.folds}-fold whole-day expanding-window CV within Jan-Aug; no shuffling.\n')
        f.write('Fold boundaries in cv_fold_dates.csv; C selected by mean fold AP.\n')
        f.write('CV mean AP: '+', '.join(f'{g}={best[g]["CV_AP"]:.4f} (C={best[g]["C"]:g})' for g in cols_map)+'\n')
        f.write('Monthly usable N and delay rate: '+monthly_str+'\n')
        f.write(summary.to_string(index=False)+'\n')
        f.write(f'M4 minus M1 validation AP = {gain:.4f}, paired day-bootstrap 95% CI [{delta_ci[0]:.4f},{delta_ci[1]:.4f}]\n')
        f.write(f'M1 threshold={threshold:.2f}, missed delay:false alarm cost={args.cost_ratio:g}:1; precision={precision_score(y,yyhat):.3f}; recall={recall_score(y,yyhat):.3f}; confusion TN={tn} FP={fp} FN={fn} TP={tp}\n')
        f.write(f'{completed} usable bootstrap draws. Random seed=42. November-December flights never scored.\n')
    is_tie=delta_ci[0]<=0<=delta_ci[1]
    conclusion=('The paired interval includes zero, so the observed gain is not distinguishable from a tie at this stage.' if is_tie
                else 'The paired interval excludes zero on validation; final confirmation is reserved for the untouched test set.')
    report=f'''# Which Flight Information Matters Most? PHL Departure Delays

## Group
{args.team}

## Changes since the proposal
We removed month from all feature sets because November and December are unseen categories in January–August training. Our primary comparison is the average-precision (AP) gain of M4 (schedule, airline, destination) over M1 (schedule only); classifier comparisons are secondary. C is selected by five expanding-window, whole-day chronological folds within January–August. September–October is reserved for checkpoint scoring and threshold selection; November–December is untouched.

## Data
Source: 2025 BTS Reporting Carrier On-Time Performance, PHL-origin flights. N={usable:,} usable flights from {raw:,} PHL records; {dropped_cancelled:,} cancelled and {dropped_missing:,} additional unusable records excluded. M1 has two inputs (scheduled departure time and day of week); the binary output is DepDel15 (delay >=15 min). Across the twelve months, monthly N and delay rate are in monthly_summary.csv.

## Split
Train: January–August (N={len(train):,}); validation: September–October (N={len(val):,}); held-out test: November–December (N={len(test):,}). Within January–August, five expanding-window folds use whole calendar days with no shuffling (seed 42). The one-hot encoder and scaler are refitted within each fold. September–October is the checkpoint validation; the test set was not evaluated.

## Baseline
One-hot logistic regression (M1). C was selected using mean AP across five chronological training-period folds from a log-spaced grid [0.001, 0.01, 0.1, 1, 10, 100]; selected C={best['M1']['C']:g}. We chose a classification threshold of {threshold:.2f} on validation, assuming a missed delay costs {args.cost_ratio:g} times a false alarm.

## Result
M1 validation AP={best['M1']['AP']:.4f}; 95% whole-day bootstrap CI [{ci['M1'][0]:.4f}, {ci['M1'][1]:.4f}] ({completed} replicates). Validation delay rate={y.mean():.3f}. Exploratory M4 validation AP={best['M4']['AP']:.4f}; paired M4-M1 gain={gain:+.4f}, 95% day-bootstrap CI [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]. {conclusion}

## Issues found
Flights on one calendar day share conditions, so we bootstrap days rather than independent flights. Categorical levels absent in training are ignored safely by the encoder. C is not tuned on September–October, but the classification threshold is. Validation AP is not the final held-out estimate; November–December remains untouched.

## Next steps
By 16 November: Jiayi will analyze the four nested feature sets and paired AP differences; Xinyu will verify logistic tuning, calibration and threshold costs; Yuxi will implement scaled/tuned KNN and a depth/leaf-tuned random forest on the same chronological splits. The team will review errors and jointly prepare the early paper draft.

## Code
{args.code_url}
'''
    words=len(report.split())
    if words>=500:print('WARNING: report exceeds 500 words:',words)
    (out/'baseline.txt').write_text(report,encoding='utf-8')
    print('\nCHECKPOINT GENERATED:',out.resolve())
    print('Words:',words)
    print('M1 validation AP',best['M1']['AP'],'CI',ci['M1'])
    print('M4-M1 gain',gain,'CI',delta_ci)
    print('WARNING: Replace placeholders in ## Group and ## Code before submission.')

if __name__=='__main__': main()
