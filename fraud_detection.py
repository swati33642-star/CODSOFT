"""
fraud_detection.py

CodSoft Internship Project: Credit Card Fraud Detection

Pipeline:
1. Load data (data/creditcard.csv)
2. Exploratory data analysis (class imbalance check, saved plots)
3. Preprocessing (scaling Time/Amount, train/test split)
4. Handle class imbalance with SMOTE (oversampling minority class)
5. Train multiple models: Logistic Regression, Random Forest, Decision Tree
6. Evaluate with precision, recall, F1, ROC-AUC, confusion matrix
7. Save the best model and evaluation plots to output/

Usage:
    python fraud_detection.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless/VS Code terminal runs
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[warning] imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("          Continuing without SMOTE (class_weight='balanced' will be used instead).\n")

DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            f"Either run 'python generate_sample_data.py' for a quick test dataset, "
            f"or download the real dataset from "
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at {path}"
        )
    return pd.read_csv(path)


def explore_data(df):
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['Class'].value_counts()}")
    fraud_pct = df["Class"].mean() * 100
    print(f"\nFraud percentage: {fraud_pct:.4f}%")
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total")

    # Class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
    plt.yscale("log")
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", bbox_inches="tight")
    plt.close()

    # Amount distribution by class
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Class", y="Amount", data=df, showfliers=False)
    plt.title("Transaction Amount by Class")
    plt.savefig(f"{OUTPUT_DIR}/amount_by_class.png", bbox_inches="tight")
    plt.close()

    print(f"\nSaved EDA plots to {OUTPUT_DIR}/")


def preprocess(df):
    df = df.copy()
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(["Amount", "Time"], axis=1)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def balance_data(X_train, y_train):
    if HAS_SMOTE:
        print("Applying SMOTE to balance training data...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
        print(f"After SMOTE:  {y_res.value_counts().to_dict()}\n")
        return X_res, y_res
    else:
        return X_train, y_train


def get_models():
    class_weight = None if HAS_SMOTE else "balanced"
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weight, random_state=42, max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=class_weight, random_state=42, n_jobs=-1
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-score (fraud class): {f1:.4f}\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{safe_name}.png", bbox_inches="tight")
    plt.close()

    return {"model": model, "auc": auc, "f1": f1, "y_proba": y_proba}


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(7, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves_comparison.png", bbox_inches="tight")
    plt.close()


def main():
    df = load_data(DATA_PATH)
    explore_data(df)

    X_train, X_test, y_train, y_test = preprocess(df)
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_bal, y_train_bal)
        results[name] = evaluate_model(name, model, X_test, y_test)

    plot_roc_curves(results, y_test)

    # Pick best model by F1-score on the fraud class (good metric for imbalanced data)
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print("=" * 60)
    print(f"BEST MODEL: {best_name} (F1={results[best_name]['f1']:.4f}, "
          f"AUC={results[best_name]['auc']:.4f})")
    print("=" * 60)

    joblib.dump(best_model, f"{OUTPUT_DIR}/best_fraud_model.pkl")
    print(f"\nSaved best model to {OUTPUT_DIR}/best_fraud_model.pkl")
    print(f"All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
