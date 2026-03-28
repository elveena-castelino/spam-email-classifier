def predict_email(text, vectorizer, model):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0][1]

    label = "Spam" if prediction == 1 else "Not Spam"
    return label, probability