# Customer Churn Prediction & Analysis

**Can we identify which telecom customers are likely to leave — before they do?**

This project analyses customer behaviour data to uncover churn patterns, builds a predictive model, and presents findings through an interactive dashboard — the kind of end-to-end analytical workflow used by DA teams at product companies.

---

## Business Problem

Customer churn is one of the most expensive problems in telecom. Acquiring a new customer costs 5–7× more than retaining an existing one. This project answers:

- Which customer segments have the highest churn rate?
- What behavioural signals predict churn before it happens?
- Which customers should retention teams prioritise?

---

## Dataset

- **Source:** Telecom customer churn dataset (public)
- **Size:** 500+ customer records
- **Key columns:** Contract type, tenure, monthly charges, tech support usage, churn status

---

## Approach

### 1. SQL Analysis (MySQL)
Queried the database to identify churn patterns by segment:
- Churn rate by contract type (month-to-month vs annual)
- Average tenure of churned vs retained customers
- Revenue at risk by customer segment

### 2. Python EDA & Modelling
- Cleaned data: handled missing values, encoded categoricals, scaled numerical features
- Explored distributions and correlations using Pandas
- Built a **Random Forest classifier** — achieved **75% accuracy, 0.69 precision, 0.70 recall**

### 3. Tableau Dashboard
Interactive dashboard showing:
- Churn rate by contract type, tenure band, and service usage
- Customer risk segmentation
- Key drivers of churn

### 4. Streamlit Web App
Live prediction tool — input customer details, get churn probability instantly.

---

## Key Findings

1. **Month-to-month contract customers churn at 3× the rate** of annual contract customers — the single strongest churn predictor
2. **Customers in their first 12 months are the highest risk** — churn rate drops sharply after year 1, suggesting onboarding experience is critical
3. **Customers without tech support are significantly more likely to churn** — bundling support services could improve retention

---

## Business Recommendation

Retention campaigns should prioritise: new customers (< 12 months tenure) on month-to-month contracts who have not enrolled in tech support. This segment represents the highest churn risk and the highest ROI for intervention.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python (Pandas, Scikit-learn) | EDA + ML model |
| MySQL | SQL analysis + querying |
| Tableau | Business dashboard |
| Streamlit | Interactive prediction app |

---

## Project Structure

```
Customer-Churn-Prediction/
├── customer_churn.py          # EDA + model training
├── streamlit_app.py           # Interactive dashboard
├── customer_churn_prediction.csv
├── MySQL/
│   ├── schema.sql
│   ├── queries.sql            # Churn analysis queries
│   └── data_import.sql
├── Tableau/
│   ├── churn_analysis.twbx
│   └── customer_insights.twbx
└── requirements.txt
```

## Run Locally

```bash
git clone https://github.com/pranav-codes55/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

*Built by Pranav R P · PES University CSE · [LinkedIn](https://www.linkedin.com/in/pranav-rp-a89635314/)*
