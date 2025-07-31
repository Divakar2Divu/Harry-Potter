import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    # === File paths ===
    file_path = "harry_potter_quiz_training_data.xlsx"  # Ensure this file is in the same directory
    model_dir = "saved_model"
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoders.pkl")
    target_encoder_path = os.path.join(model_dir, "target_encoder.pkl")

    # === Create model directory if it doesn't exist ===
    os.makedirs(model_dir, exist_ok=True)

    # === Load data ===
    try:
        training_df = pd.read_excel(file_path, sheet_name="answers_training_data")
    except Exception as e:
        print(f"❌ Failed to load Excel file: {e}")
        return

    feature_cols = ["A1", "A2", "A3", "A4", "A5"]
    label_encoders = {}

    # === Encode features ===
    for col in feature_cols:
        le = LabelEncoder()
        training_df[col] = le.fit_transform(training_df[col])
        label_encoders[col] = le

    # === Encode target ===
    target_le = LabelEncoder()
    training_df["Character"] = target_le.fit_transform(training_df["Character"])

    # === Train model ===
    X = training_df[feature_cols]
    y = training_df["Character"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # === Save model and encoders ===
    try:
        joblib.dump(model, model_path)
        joblib.dump(label_encoders, encoder_path)
        joblib.dump(target_le, target_encoder_path)
        print("✅ Model and encoders saved successfully.")
    except Exception as e:
        print(f"❌ Error saving model or encoders: {e}")

if __name__ == "__main__":
    main()
