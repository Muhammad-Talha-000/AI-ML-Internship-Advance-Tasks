# 🧠 Advanced AI/ML Internship Tasks

This repository showcases a collection of advanced tasks completed during my **AI/ML Engineering Internship** at **DevelopersHub Corporation**. Each task explores a unique machine learning application, including **natural language processing**, **classification**, and **multimodal regression**.

## 📁 Task Summary

| Task No. | Task Title                                 | Model Used                              | Task Type      | Output                      |
| -------- | ------------------------------------------ | --------------------------------------- | -------------- | --------------------------- |
| Task 1   | News Topic Classification with BERT        | BERT (Transformers)                     | Classification | News category (0–3)         |
| Task 2   | Telco Customer Churn Prediction            | Random Forest Classifier                | Classification | Churn: Yes / No             |
| Task 3   | House Price Prediction (Images + Features) | CNN + Dense Neural Network (Multimodal) | Regression     | Predicted House Price (USD) |

---

## 📰 Task 1: News Topic Classification with BERT

### 🎯 Objective

Build a text classifier using BERT to identify the topic of a given news headline.

### 📊 Dataset

* **Source:** [AG News Dataset – Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
* **Classes:** World, Sports, Business, Sci/Tech
* **Split:** 2,000 training samples, 500 test samples

### 🛠️ Model Details

* Pre-trained `bert-base-uncased` from HuggingFace
* Fine-tuned using PyTorch
* Training config: 1 epoch, batch size 8, max length 128, mixed-precision (`fp16`)

### 📈 Performance

| Metric   | Score |
| -------- | ----- |
| Accuracy | \~89% |
| F1 Score | \~88% |

### 🌐 Live Demo (Gradio)

* **Input:** Custom news headline
* **Output:** Predicted category (e.g., Business, Sci/Tech)

---

## 📊 Task 2: Telco Customer Churn Prediction

### 🎯 Objective

Predict whether a customer is likely to churn based on service usage and demographic data.

### 📊 Dataset

* **Source:** [Telco Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Size:** 7,043 records
* **Features:** Customer demographics, billing, contracts, service plans
* **Target:** Churn (Yes/No)

### 🛠️ Pipeline

* Preprocessing: `ColumnTransformer` with pipelines for numeric and categorical features
* Model: `RandomForestClassifier`
* Hyperparameter tuning via `GridSearchCV`

### 📈 Evaluation

| Metric             | Result            |
| ------------------ | ----------------- |
| Accuracy           | \~82%             |
| Precision / Recall | ✔️                |
| Confusion Matrix   | ✔️                |
| Feature Importance | ✔️ (Top 20 shown) |

### 🔍 Sample Prediction

```python
Churn Prediction: YES
Churn Probability: 0.74
```

---

## 🏠 Task 3: Housing Price Prediction (Multimodal)

### 🎯 Objective

Estimate house prices using a **multimodal approach** that combines both image and tabular data.

### 📊 Dataset

* **Images:** `socal_pics/` (15,000 images)
* **Tabular Data:** `listings.csv`
* **Features:** Bedrooms, bathrooms, square footage, neighborhood
* **Target:** House price (USD)

### 🧠 Model Architecture

* **Tabular Branch:** Dense neural layers
* **Image Branch:** CNN (input size: 224x224)
* **Fusion:** Merged into a single regression head
* **Training:** 2,000 samples, 5 epochs

### 📈 Evaluation

| Metric        | Value         |
| ------------- | ------------- |
| MAE           | \~60K–80K USD |
| Val MAE       | \~64K–85K USD |
| Training Loss | Stable        |

### 💡 Key Insights

* Tabular features (especially `sqft`) were highly influential
* Images contributed to understanding architectural style and condition
* The combined model outperformed the tabular-only baseline

---

## 🚀 Running the Tasks

### 🔧 Setup Instructions

Clone the repository:

```bash
git clone https://github.com/Abdul-Wahab1010/AI-ML-Engineering-Internship-Tasks.git
cd AI-ML-Engineering-Internship-Tasks
```

Navigate into any task folder:

```bash
cd Task_01-News-Topic-Classifier
# or
cd Task_02-Telco-Churn-Prediction
# or
cd Task_03-Housing-Price-Prediction
```

Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tensorflow opencv-python gradio joblib
```

Launch the Jupyter notebook:

```bash
jupyter notebook <TASK_NOTEBOOK>.ipynb
```

---

## 👤 Author

**Abdul Wahab**
AI/ML Engineering Intern – DevelopersHub Corporation
GitHub: [@Abdul-Wahab1010](https://github.com/Abdul-Wahab1010)

---

Let me know if you want a version in PDF or formatted for a portfolio website.
