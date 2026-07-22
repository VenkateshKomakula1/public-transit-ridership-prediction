# Flagstaff Mountain Line Transit Boarding Prediction

Graduate capstone project (BAN 586, M.S. Business/Data Analytics, Northern
Arizona University, Spring 2024): predicting Automatic Passenger Count (APC)
boarding numbers for a public bus service using route, timing, and historical
ridership data, with the goal of informing schedule and route planning.

**Full written report:** [`docs/Flagstaff_Mountain_Line_Capstone_Report.pdf`](docs/Flagstaff_Mountain_Line_Capstone_Report.pdf)
**Notebook:** [`notebooks/flagstaff_transit_boarding_model.ipynb`](notebooks/flagstaff_transit_boarding_model.ipynb)

## Problem

The Flagstaff Mountain Line wanted to understand what drives passenger
boarding counts on a given trip, so schedules and route investment could be
aligned with actual demand rather than fixed assumptions. The dataset
included trip-level records (route number, block number, GPS coordinates,
trip start/end time, on-time performance) and the APC boarding count as the
prediction target.

## Approach

1. **Baseline model** — Linear Regression on route number, block number,
   latitude, and longitude.
2. **Model complexity** — Random Forest Regressor on the same baseline
   features, to test whether a non-linear model alone would help.
3. **Feature engineering** — added `hourOfDay` and `dayOfWeek` (extracted
   from trip start time) and `routeAvgBoardingCount` (historical average
   boarding count per route), then retrained.
4. **Data enrichment** — merged in a local events dataset (encoded as
   categorical dummies) to test whether external context improved
   predictions further.

## Results

| Model | Features | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | baseline (route, block, lat/long) | 4.16 | 2.04 | -0.08 |
| Random Forest | baseline | 7.79 | 2.79 | -1.03 |
| Random Forest | + time features + route popularity | 5.15 | 2.27 | -0.34 |
| Random Forest | + local events data | 0.87 | 0.93 | -0.28 |

RMSE improved substantially as features were added (roughly 2 passengers of
average error down to under 1), but R² stayed negative throughout — the
models never explained more variance than a simple mean-based prediction
would. Feature importance analysis (see notebook) pointed to time-of-day and
route popularity as the strongest signals available in this dataset, but the
dataset itself was small (100 trips) and the relationship between the
available features and boarding counts appears more complex than these
models captured.

**Honest takeaway, stated directly in the report:** this is a real,
representative case of applied ML on a small, real-world dataset — the value
is in the rigorous iteration (baseline → complexity → feature engineering →
enrichment) and the diagnosis of *why* each step did or didn't help, not in
a flattering final metric. That diagnostic reasoning is documented in full in
the report's Results and Recommendations section.

## Repo structure

```
├── notebooks/
│   └── flagstaff_transit_boarding_model.ipynb   # full modeling workflow, with outputs
├── docs/
│   └── Flagstaff_Mountain_Line_Capstone_Report.pdf  # full written report
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/flagstaff_transit_boarding_model.ipynb
```

**Note on data:** the raw trip-level dataset (`final_merged_data.csv`) was
provided by the Flagstaff Mountain Line under an academic data-sharing
arrangement and is not included in this repository. The notebook is included
with its original outputs so the full analysis is visible without needing
the raw file.

## Tech stack

Python, pandas, scikit-learn (Linear Regression, Random Forest), NumPy,
Matplotlib.

## Author

Venkatesh Komakula
<!-- Add your LinkedIn / portfolio link here before pushing -->

