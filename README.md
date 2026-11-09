# ESE 5420 — PHL Departure Delay Prediction

Course project baseline checkpoint: predicting whether a **non-cancelled 2025 domestic flight departing from Philadelphia International Airport (PHL)** is delayed at least 15 minutes (`DepDel15`). Data source: [U.S. BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ).

## Files

- `PHL_Reproduce_Results.ipynb` — Jupyter notebook to reproduce the checkpoint.
- `PHL_checkpoint_auto.py` — data preparation, chronological split, logistic-regression training/tuning, day-block bootstrap, and plots.
- `PHL_2025_raw_departures.csv` — 2025 PHL-origin BTS flight records, including cancellations for filtering.
- `baseline.png` — validation precision–recall comparison.
- `validation_results.csv` — nested feature-set validation results.
- `monthly_summary.csv` — monthly sample sizes and delay rates.

## Reproduce

Place the notebook, Python script and CSV in the same directory. Run all cells in `PHL_Reproduce_Results.ipynb`. Alternatively:

```bash
pip install pandas numpy scikit-learn matplotlib requests
python PHL_checkpoint_auto.py --data PHL_2025_raw_departures.csv --out rerun_checkpoint --bootstrap 400
```

The script filters `Origin == PHL`, removes cancellations and unusable records, and selects logistic-regression regularization using September–October validation. It trains on January–August 2025, and **does not use November–December 2025 for training, tuning or checkpoint scoring**. A fixed random seed of 42 is used. Bootstrap intervals resample whole calendar days to account for within-day dependence.

Primary project question: average-precision improvement of schedule + airline + destination (M4) over schedule-only (M1), with model-method comparisons planned next. For the current checkpoint, displayed model scores are **validation results, not final held-out test performance**.
