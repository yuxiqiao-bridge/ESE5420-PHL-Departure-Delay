# ESE 5420 — PHL Departure Delay Prediction (5-fold CV)

**Research question:** How much do airline and destination improve average precision (AP) for predicting departures delayed by at least 15 minutes at PHL?

## Reproduce
Run `PHL_Reproduce_Results.ipynb` in Google Colab (its setup cell clones this public repository if needed), or run in a terminal:

```bash
pip install pandas numpy scikit-learn matplotlib requests
python PHL_checkpoint_auto.py --data PHL_2025_raw_departures.csv --out rerun_checkpoint --bootstrap 400 --folds 5 --team "5420 Group; Yuxi, Jiayi, Xinyu" --code-url "https://github.com/yuxiqiao-bridge/ESE5420-PHL-Departure-Delay"
```

## Data, tuning, and evaluation
Source: [U.S. BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ). The repository's `PHL_2025_raw_departures.csv` contains PHL-origin flights from 2025, including cancellations. After filtering cancellations and unusable records, 95,480 flights remain. The binary target is `DepDel15`.

Training uses January–August. We tune each logistic-regression model's `C` using **five expanding-window chronological validation folds**, partitioned by whole calendar days within January–August. No shuffling. The regularization grid is `[0.001, 0.01, 0.1, 1, 10, 100]`; the criterion is mean fold AP. The numeric scaler and categorical one-hot encoder are refit separately inside each training fold. Month is not a predictor.

The selected models refit on all January–August flights and are scored on September–October, which also determines the cost-based classification threshold (missed delay:false alarm = 5:1). Average precision is threshold-free. Uncertainty intervals are computed from 400 bootstrap draws resampling whole calendar days. November–December is retained as an untouched final test period (only counts are tabulated).

## Results (September–October validation; not held-out test)
M1 AP = 0.2778, whole-day bootstrap 95% CI [0.2460, 0.3142]; M4 AP = 0.2901. Paired M4 minus M1 AP gain = +0.0122, 95% CI [+0.0076, +0.0177].

Relevant files: `PHL_checkpoint_auto.py`, `PHL_Reproduce_Results.ipynb`, `validation_results.csv`, `CV_summary.csv`, `cv_fold_dates.csv`, `C_validation_grid.csv`, `baseline.png`. The `baseline.png` figure should be generated from the current script to show current 5-fold results, rather than reused from the previous non-CV run.
