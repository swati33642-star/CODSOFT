import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

# Load dataset
df = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "description"],
    skiprows=1
)

print("Dataset loaded successfully!")

# Remove missing values
df = df.dropna()

# Use a sample for faster training
df = df.sample(n=10000, random_state=42)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Features and Target
X = df["description"]
y = df["genre"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nCreating TF-IDF vectors...")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("TF-IDF completed!")

print("\nTraining model...")

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")
