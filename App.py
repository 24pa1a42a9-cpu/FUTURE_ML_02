import joblib

model = joblib.load("ticket_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

while True:

    ticket = input("Enter Ticket: ")

    vector = vectorizer.transform([ticket])

    prediction = model.predict(vector)

    print("Predicted Category:", prediction[0])