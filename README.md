# 🩺 Diabetes Prediction System (Logistic Regression)

A machine learning–based web application that predicts **diabetes risk** using patient health data.
The system uses **Logistic Regression**, probability-based predictions, and **threshold tuning** to provide **Low, Moderate, or High risk** outcomes.


## 📌 Project Overview

This project aims to predict whether a person is at risk of diabetes based on medical and lifestyle features such as age, BMI, HbA1c level, blood glucose level, hypertension, heart disease, gender, and smoking history.

Instead of a simple yes/no prediction, the model outputs a **risk probability** and classifies users into **risk bands**, making the system more realistic and healthcare-friendly.


## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Joblib


## 📊 Dataset Information

* **Rows:** 100,000
* **Columns:** 16
* **Target Variable:** `diabetes` (0 = No, 1 = Yes)

### Key Features:

* Age
* BMI
* HbA1c level
* Blood glucose level
* Hypertension
* Heart disease
* Gender
* Smoking history


## 🧹 Data Preprocessing

The dataset was cleaned and prepared using the following steps:

* Removed duplicates and handled missing values
* Treated outliers (especially BMI) using statistical methods
* Encoded categorical variables using **One-Hot Encoding**
* Dropped reference categories to avoid **dummy variable trap**
* Scaled numerical features using **StandardScaler**
* Ensured no data leakage using proper train-test split


## 🤖 Model Used

**Logistic Regression**

### Model Configuration:

* Solver: `lbfgs`
* Regularization: `L2`
* Max Iterations: `1000`
* Class Weight: `balanced`

This configuration works efficiently for large datasets and imbalanced classes.


## 📈 Model Performance

### Evaluation Metrics:

* **Accuracy:** 0.89
* **Recall (Diabetes):** 0.88
* **ROC–AUC Score:** **0.96**

The high ROC-AUC score indicates excellent class separation capability.


## 🎯 Threshold Tuning (Key Highlight)

Instead of using the default threshold (0.5), the decision threshold was **tuned** to improve precision and control false positives.

### Risk Classification Logic:

* **Probability ≥ 0.40** → High Diabetes Risk
* **0.30 < Probability < 0.40** → Moderate (Borderline) Risk
* **Probability ≤ 0.30** → Low Risk

This approach makes the model more suitable for healthcare screening scenarios.


## 🌐 Web Application (Streamlit)

The trained model is deployed as a **Streamlit web app** that:

* Takes user-friendly health inputs
* Converts inputs into model-compatible format
* Displays diabetes risk probability
* Shows risk level (Low / Moderate / High)
* Uses tuned threshold logic

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Run the Streamlit app
streamlit run app.py



## 📁 Project Structure

diabetes-prediction-system/
│
├── data/
│   ├── raw/
│   │   └── diabetes_prediction.csv
│   │
│   └── processed/
│       └── cleaned_data.csv
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── notebook/
│   ├── 01_Understanding_Dataset.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   └── 04_Training_Model.ipynb
│
├── venv/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


## 🧠 Key Learnings

* Importance of proper data preprocessing
* Handling class imbalance using class weights
* Why accuracy alone is not sufficient for medical ML
* How threshold tuning improves real-world usability
* Building an end-to-end ML pipeline from data to deployment


## ⚠️ Disclaimer

This project is for **educational purposes only** and should **not be used as a medical diagnosis tool**. Always consult a qualified healthcare professional for medical decisions.


## 🚀 Future Improvements

* Add SHAP / feature contribution explanation
* Enable CSV bulk prediction
* Improve UI with charts and confidence indicators
* Deploy on Streamlit Cloud


## 👤 Author

**Amaan Behlim**
CSE (AI/ML) Student | Machine Learning Enthusiast
