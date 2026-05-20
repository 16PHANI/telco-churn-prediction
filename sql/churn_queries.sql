-- ============================================================
-- churn_queries.sql
-- Customer Churn Analysis — SQL (PostgreSQL)
-- Dataset: telco_churn (7,043 records)
-- ============================================================

-- ── 1. Overall churn rate ────────────────────────────────────
SELECT
    "Churn",
    COUNT(*)                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM telco_churn
GROUP BY "Churn"
ORDER BY "Churn";


-- ── 2. Churn rate by tenure cohort (KEY INSIGHT: 38% in 0-3 mo) ──
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND  3  THEN '0-3 months'
        WHEN tenure BETWEEN 4  AND 12  THEN '4-12 months'
        WHEN tenure BETWEEN 13 AND 24  THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48  THEN '25-48 months'
        ELSE '49-72 months'
    END                                            AS cohort,
    COUNT(*)                                       AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        1
    )                                              AS churn_rate_pct
FROM telco_churn
GROUP BY 1
ORDER BY MIN(tenure);


-- ── 3. Churn rate by contract type ──────────────────────────
SELECT
    "Contract",
    COUNT(*)                                       AS total,
    ROUND(
        AVG(CASE WHEN "Churn" = 'Yes' THEN 100.0 ELSE 0 END),
        1
    )                                              AS churn_rate_pct
FROM telco_churn
GROUP BY "Contract"
ORDER BY churn_rate_pct DESC;


-- ── 4. Churn rate by payment method ─────────────────────────
SELECT
    "PaymentMethod",
    COUNT(*)                                       AS total,
    ROUND(
        AVG(CASE WHEN "Churn" = 'Yes' THEN 100.0 ELSE 0 END),
        1
    )                                              AS churn_rate_pct
FROM telco_churn
GROUP BY "PaymentMethod"
ORDER BY churn_rate_pct DESC;


-- ── 5. Avg monthly charges — churn vs retained ──────────────
SELECT
    "Churn",
    ROUND(AVG("MonthlyCharges"), 2)               AS avg_monthly_charges,
    ROUND(STDDEV("MonthlyCharges"), 2)            AS std_monthly_charges,
    ROUND(MIN("MonthlyCharges"), 2)               AS min_charges,
    ROUND(MAX("MonthlyCharges"), 2)               AS max_charges
FROM telco_churn
GROUP BY "Churn";


-- ── 6. Tech support impact on churn (internet customers only) ──
SELECT
    "TechSupport",
    COUNT(*)                                       AS total,
    ROUND(
        AVG(CASE WHEN "Churn" = 'Yes' THEN 100.0 ELSE 0 END),
        1
    )                                              AS churn_rate_pct
FROM telco_churn
WHERE "InternetService" != 'No'
GROUP BY "TechSupport"
ORDER BY churn_rate_pct DESC;


-- ── 7. Churn rate by internet service type ───────────────────
SELECT
    "InternetService",
    COUNT(*)                                       AS total,
    ROUND(
        AVG(CASE WHEN "Churn" = 'Yes' THEN 100.0 ELSE 0 END),
        1
    )                                              AS churn_rate_pct
FROM telco_churn
GROUP BY "InternetService"
ORDER BY churn_rate_pct DESC;


-- ── 8. Senior citizen churn rate ────────────────────────────
SELECT
    CASE WHEN "SeniorCitizen" = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*)                                       AS total,
    ROUND(
        AVG(CASE WHEN "Churn" = 'Yes' THEN 100.0 ELSE 0 END),
        1
    )                                              AS churn_rate_pct
FROM telco_churn
GROUP BY "SeniorCitizen"
ORDER BY churn_rate_pct DESC;


-- ── 9. Null / data quality check ────────────────────────────
SELECT
    COUNT(*)                                         AS total_rows,
    SUM(CASE WHEN "customerID"     IS NULL THEN 1 ELSE 0 END) AS null_customerID,
    SUM(CASE WHEN "TotalCharges"   IS NULL THEN 1 ELSE 0 END) AS null_TotalCharges,
    SUM(CASE WHEN "MonthlyCharges" IS NULL THEN 1 ELSE 0 END) AS null_MonthlyCharges,
    SUM(CASE WHEN "tenure"         IS NULL THEN 1 ELSE 0 END) AS null_tenure,
    SUM(CASE WHEN "Churn"          IS NULL THEN 1 ELSE 0 END) AS null_Churn
FROM telco_churn;


-- ── 10. Revenue at risk from churned customers ───────────────
SELECT
    "Churn",
    COUNT(*)                                       AS customers,
    ROUND(SUM("MonthlyCharges"), 2)               AS total_monthly_revenue,
    ROUND(AVG("MonthlyCharges"), 2)               AS avg_monthly_revenue
FROM telco_churn
GROUP BY "Churn";


-- ── 11. High-value churned customers (top retention targets) ──
SELECT
    "customerID",
    tenure,
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService"
FROM telco_churn
WHERE "Churn" = 'Yes'
  AND "MonthlyCharges" > 70
ORDER BY "MonthlyCharges" DESC
LIMIT 20;
