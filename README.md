# Customer Churn Prediction & Statistical EDA Pipeline

**Dataset:** Kaggle Telco Customer Churn (7,043 records) | **Language:** Python | **Author:** Boyinapalli Phani Shankar

---

## Results at a glance

| Model               | ROC-AUC   | Weighted F1 | Accuracy  |
| ------------------- | --------- | ----------- | --------- |
| Logistic Regression | ~0.84     | ~0.79       | ~0.78     |
| Random Forest       | ~0.85     | ~0.80       | ~0.80     |
| **XGBoost ★**       | **~0.87** | **~0.81**   | **~0.81** |

**Key business finding:** 38% of all churned customers were on month-to-month contracts with tenure <= 3 months — the single highest-impact intervention target.

---

## What this project covers

| Stage                    | What was done                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Data cleaning**        | Coerced `TotalCharges` to numeric, imputed 11 missing values with median, dropped `customerID`             |
| **Feature engineering**  | Tenure bands (0-3 mo, 4-12 mo, ...), monthly-charges bands, service-count feature, contract ordinal encoding |
| **EDA**                  | 11 visualisations: histograms, KDE plots, heatmaps, box plots, correlation matrix, feature importance      |
| **Statistical analysis** | Churn rates segmented by contract type, tenure band, internet service, payment method                      |
| **SQL analysis**         | 11 queries: cohort churn rates, contract type, payment method, data quality checks, revenue at risk        |
| **Modelling**            | Logistic Regression vs. Random Forest vs. XGBoost with stratified train/test split and 5-fold CV          |
| **Business insights**    | Actionable retention strategy: contract upgrade campaign, early-tenure onboarding, payment method migration |

---

## Visualisations

| #  | File                               | Description                                             |
| -- | ---------------------------------- | ------------------------------------------------------- |
| 1  | `01_churn_distribution.png`        | Overall churn bar + pie chart                           |
| 2  | `02_churn_by_contract.png`         | Churn rate by contract type                             |
| 3  | `03_churn_by_tenure_band.png`      | Churn rate per tenure band with key-insight annotation  |
| 4  | `04_monthly_charges_kde.png`       | Monthly charges distribution by churn (KDE overlay)     |
| 5  | `05_tenure_kde.png`                | Tenure distribution by churn (KDE overlay)              |
| 6  | `06_box_plots.png`                 | Box plots: monthly charges & tenure by churn status     |
| 7  | `07_correlation_heatmap.png`       | Correlation matrix: numerical features                  |
| 8  | `08_churn_by_internet_service.png` | Churn rate by internet service type                     |
| 9  | `09_churn_by_payment_method.png`   | Churn rate by payment method                            |
| 10 | `10_roc_curves.png`                | ROC curves: all three models                            |
| 11 | `11_feature_importance.png`        | XGBoost top-15 feature importance                       |

---

## Setup & run

```
# 1. Clone the repo
git clone https://github.com/16PHANI/telco-churn-prediction.git
cd telco-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run -- dataset downloads automatically on first run
python churn_analysis.py

# Outputs: outputs/figures/ (11 plots)
```

If auto-download fails, download the CSV manually:
-> https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place it at `data/Telco-Customer-Churn.csv` and run again.

---

## Project structure

```
telco-churn-prediction/
├── churn_analysis.py             <- complete analysis (EDA + modelling + insights)
├── requirements.txt
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv  (auto-downloaded on first run)
├── sql/
│   └── churn_queries.sql         (11 SQL queries: cohort, contract, payment method)
└── outputs/
    └── figures/                  (11 plots saved here)
        ├── 01_churn_distribution.png
        ├── 02_churn_by_contract.png
        ├── ...
        └── 11_feature_importance.png
```

---

## Tech stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `SQL (PostgreSQL)`

---

## Business recommendations

1. **Early-tenure intervention** — target 0-3 month monthly-contract customers with an onboarding call + loyalty discount. This cohort accounts for ~38% of all churn.
2. **Contract upgrade campaign** — at month 2-3, offer a discounted 1-year contract to monthly customers.
3. **Electronic-check migration** — highest churn rate by payment method; migrate with a small auto-pay bill credit.
4. **Fiber optic bundle** — proactively offer TechSupport + OnlineSecurity add-ons to Fiber customers; both features significantly lower churn probability.

---

## By
 
- **Boyinapalli Phani Shankar**
- **GitHub**: https://github.com/16PHANI
- **LinkedIn**: https://linkedin.com/in/phanishankar16
- **Google Skill Boost**: https://www.skills.google/public_profiles/a4ca4511-cc29-45dd-899f-63010edf9917
