# 📧 Spam Email Classifier

## 🚀 Project Overview

This project builds a machine learning model to classify emails as **Spam** or **Not Spam** using **TF-IDF vectorization** and **Logistic Regression**.

---

## 🧠 Approach

* Text converted into numerical features using TF-IDF
* Logistic Regression used for classification
* N-grams used to capture word combinations

---

## ⚙️ Features

* Real-time spam prediction
* Model interpretability (top spam words)
* Clean and modular code structure

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* Confusion Matrix

---

## 🧪 Example

Input:
"Congratulations! You won a free prize"

Output:
Spam (0.92 confidence)

---

## 📁 Project Structure

spam-classifier/
│
├── data/
├── src/
├── app.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧩 Tech Stack

* Python
* Scikit-learn
* Pandas
* Streamlit

---

## 📌 Future Improvements

* Deep learning models
* Deployment on cloud
* Better text preprocessing
