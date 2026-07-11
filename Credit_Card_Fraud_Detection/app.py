"""
app.py

Streamlit website for the Credit Card Fraud Detection project.

Run this AFTER you've run fraud_detection.py at least once (it needs
output/best_fraud_model.pkl and output/preprocessing.pkl to exist).

Usage:
    streamlit run app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

MODEL_PATH = "output/best_fraud_model.pkl"
PREPROCESS_PATH = "output/preprocessing.pkl"

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocess = joblib.load(PREPROCESS_PATH)
    return model, preprocess


def prepare_features(df, preprocess):
    """Apply the same scaling used during training, and align columns."""
    df = df.copy()
    df["Amount_scaled"] = preprocess["amount_scaler"].transform(df[["Amount"]])
    df["Time_scaled"] = preprocess["time_scaler"].transform(df[["Time"]])
    df = df.drop(["Amount", "Time"], axis=1)
    # Ensure columns are in the exact order the model was trained on
    df = df.reindex(columns=preprocess["feature_order"], fill_value=0)
    return df


def main():
    st.title("💳 Credit Card Fraud Detection")
    st.caption("CodSoft Internship Project — Logistic Regression / Decision Tree / Random Forest")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESS_PATH)):
        st.error(
            "Model files not found. Please run `python fraud_detection.py` first "
            "in your terminal — it creates the files this app needs "
            "(output/best_fraud_model.pkl and output/preprocessing.pkl)."
        )
        return

    model, preprocess = load_artifacts()

    tab1, tab2 = st.tabs(["📁 Upload CSV (batch check)", "✍️ Manual entry (single transaction)"])

    # ---------------- TAB 1: CSV upload ----------------
    with tab1:
        st.write(
            "Upload a CSV with the same columns as the training data "
            "(`Time`, `V1`...`V28`, `Amount`) — no `Class` column needed."
        )
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            has_class = "Class" in raw_df.columns
            feature_df = raw_df.drop(columns=["Class"]) if has_class else raw_df.copy()

            try:
                X = prepare_features(feature_df, preprocess)
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                result_df = raw_df.copy()
                result_df["Predicted"] = np.where(preds == 1, "Fraud", "Legit")
                result_df["Fraud_Probability"] = np.round(probs, 4)

                n_fraud = int((preds == 1).sum())
                col1, col2, col3 = st.columns(3)
                col1.metric("Total transactions", len(result_df))
                col2.metric("Flagged as fraud", n_fraud)
                col3.metric("Fraud rate flagged", f"{n_fraud / len(result_df) * 100:.2f}%")

                st.dataframe(
                    result_df.sort_values("Fraud_Probability", ascending=False),
                    use_container_width=True,
                )

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV", csv_out, "fraud_predictions.csv", "text/csv"
                )
            except Exception as e:
                st.error(f"Couldn't process this file: {e}")

    # ---------------- TAB 2: Manual entry ----------------
    with tab2:
        st.write(
            "Enter values for a single transaction. `V1`-`V28` are anonymized "
            "PCA features from the original dataset — leave them at 0 if you're "
            "just testing the interface, since their real-world meaning isn't public."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            time_val = st.number_input("Time (seconds since first transaction)", value=0.0)
        with col_b:
            amount_val = st.number_input("Amount ($)", value=100.0, min_value=0.0)

        st.markdown("**V1 – V28 (PCA features)**")
        v_values = {}
        cols = st.columns(7)
        for i in range(1, 29):
            col = cols[(i - 1) % 7]
            v_values[f"V{i}"] = col.number_input(f"V{i}", value=0.0, key=f"v{i}", label_visibility="visible")

        if st.button("Check this transaction", type="primary"):
            row = {**v_values, "Time": time_val, "Amount": amount_val}
            input_df = pd.DataFrame([row])
            X = prepare_features(input_df, preprocess)
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0, 1]

            if pred == 1:
                st.error(f"⚠️ Likely FRAUD — probability {prob * 100:.2f}%")
            else:
                st.success(f"✅ Likely legitimate — fraud probability {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
