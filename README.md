# 💳 High-Performance Financial Fraud Detection System

## 🎯 Project Goal
The core objective was to build a **machine learning model** capable of accurately identifying **fraudulent transactions** within a large, heavily imbalanced dataset (**6.3 million** records with only **0.13%** fraud).  
The final system was designed to **minimize financial loss** (high Recall) while maintaining a **low false alarm rate** (high Precision).

---

## 🚀 Final Performance Metrics
The tuned **Random Forest Pipeline** achieved the following robust results on the **1.9 million** record test set:

| Metric | Score | Impact |
|:--|:--|:--|
| **F1-Score (Fraud)** | **0.8954** | The final, balanced success metric. |
| **Precision** | **0.8892 (~89%)** | Only 11% of alerts are false alarms — operationally viable. |
| **Recall** | **0.9018 (~90%)** | Catches 9 out of 10 actual fraud cases. |
| **PR-AUC** | **0.9631** | Most honest measure of imbalanced performance — near-perfect discrimination. |

---

## 💡 Technical Strategy: Conquering Imbalance

### 1. 🧮 Feature Engineering: Integrity Checks
The biggest lift came from creating **domain-specific features** based on the principle of *conservation of money*.

- **Balance Integrity:**  
  Engineered two key variables:
  - `balanceDiffOrig`
  - `balanceDiffDest`  
  These measure deviations between expected and actual account balances after a transaction — highly predictive of manipulation.

- **Data Preparation:**  
  Stabilized the severely skewed transaction amounts using the `log(x + 1)` transform to reduce the impact of outliers.

---

### 2. 🌲 Model Selection: Random Forest
Initial tests with **Logistic Regression** (baseline) failed — achieving a Precision of only **2.26%** and producing **100,000+ false positives**, despite high Recall.

The **Random Forest Classifier** was chosen for its ability to model **complex, non-linear interactions** between features.  
Both models used `class_weight='balanced'` to ensure attention to the rare fraud class.

---

### 3. 🎯 Threshold Optimization: The Game Changer
The default **0.5 threshold** was unusable for this problem.

**Methodology:**
- Switched from `.predict()` to `.predict_proba()`
- Calculated F1-score across thresholds from **0.0 → 1.0**
- Found optimal cutoff at **0.27**

**Result:**  
Lowering the decision threshold to `0.27` converted high-probability scores (e.g. 0.40) into “Fraud” predictions — significantly increasing **Precision** while maintaining high **Recall**.  
This was visually validated via the **F1-score vs. Threshold** tuning curve.

---

## 🛠️ Project Structure and Deployment

The entire workflow is built with **Scikit-learn’s `Pipeline`** and **`ColumnTransformer`** for a **clean, reproducible, and leak-free** process.

**Preprocessing Pipeline:**
- `StandardScaler` → Numeric features  
- `OneHotEncoder` → Categorical features (e.g., `type`)  
- Automatically applied to new data during prediction.

**Model Persistence:**
- Final fitted pipeline saved as **`fraud_detection_rf.pkl`** using `joblib`
- Enables fast, reliable deployment.

**Demo Application:**
- Deployed via **Streamlit** (`app.py`)
- Allows users to input transaction details and instantly view:
  - Fraud **probability**
  - Final **verdict** (based on optimized 0.27 threshold)

---

## ⚙️ How to Run

```bash
# 1️⃣ Install dependencies
pip install pandas scikit-learn numpy joblib streamlit

# 2️⃣ Run the live demo
streamlit run app.py
