import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from src.preprocess import get_vectorizer, transform_data
from src.train import train_model

# Load data
data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Convert labels
data['label'] = (data['label'] == 'spam').astype(int)

# Split
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
vectorizer = get_vectorizer()
X_train_tfidf, X_test_tfidf = transform_data(vectorizer, X_train, X_test)

# Train model
model = train_model(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully.")