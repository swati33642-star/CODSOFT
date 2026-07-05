import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

while True:
    message = input("Enter SMS: ")

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Spam Message")
    else:
        print("Legitimate Message")