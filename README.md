# 🧠 Customer Churn & Lifetime Value Prediction

This is an end-to-end machine learning project to simulate and solve a real-world customer analytics problem. The project involves **predicting customer churn (classification)**, **forecasting customer lifetime value (regression)**, and **segmenting users (clustering)** using a synthetic customer dataset.

---

## 🚀 Project Objectives

- Predict whether a customer will **churn** or not
- Predict **LTV (Lifetime Value)** of a customer
- Cluster customers into natural **behavioral segments**
- Interpret model decisions for business transparency

---

## 📁 Project Structure (Phases)

| Phase | Description |
|-------|-------------|
| **1. EDA** | Analyzed distributions, imbalances, outliers, temporal patterns |
| **2. Preprocessing** | Missing values, encoding, scaling, tenure extraction |
| **3. Feature Engineering** | RFM metrics, log transforms, transaction summaries |
| **4. Classification** | Models: Logistic Regression, Random Forest, XGBoost |
| **5. Regression** | LTV prediction with log transform + tree models |
| **6. Clustering** | PCA + KMeans to discover customer segments |
| **7. Tuning** | Hyperparameter tuning with `RandomizedSearchCV` |
| **8. Explainability** | Feature importances, SHAP (optional) |

---

## 📊 Datasets Used

- `customers.csv` → demographics + signup info  
- `interactions.csv` → complaints, satisfaction, tickets  
- `transactions.csv` → purchases, timestamps, categories  
- `campaigns.csv` → marketing click/purchase behavior

---

## 🧠 ML Models Used

### 🔹 Classification (Churn)
- Logistic Regression
- Random Forest Classifier ✅
- XGBoost Classifier

**Best Results:**
- Precision: `~0.81`
- Recall: `~0.45`
- ROC-AUC: `~0.72`

---

### 🔸 Regression (LTV)
- Linear Regression
- Random Forest Regressor ✅
- XGBoost Regressor

**Best Results (after log-transform):**
- MAE: `~1.32`
- RMSE: `~5.48`
- R² Score: `~0.99998`

---

### 🔺 Clustering (Unsupervised)
- PCA for dimensionality reduction
- KMeans (k=3)
- Customer profiling using grouped means

---

## 🔍 Key Insights

- **Satisfaction score & complaint count** were top churn drivers
- **Log-transforming LTV** greatly stabilized regression performance
- Customers clustered into:
  - Passive but loyal users
  - Highly engaged but risky churners
  - Average customers

---

## ⚙️ Tools & Libraries

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- XGBoost
- SHAP (optional)
- Jupyter Notebook

---

## 📈 Next Steps
- Add SHAP explainability for regression
- Deploy model with `Flask` or `Streamlit`
- Visual dashboard with Tableau or Plotly


