import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import cohen_kappa_score
from modul_pengecekan_tatabahasa.fitur_tata_bahasa import extract_grammar_features

# === Fungsi QWK ===
def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=10):
    y_true = np.clip(np.round(y_true), min_rating, max_rating).astype(int)
    y_pred = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def evaluate_qwk(y_true, y_pred):
    score = quadratic_weighted_kappa(y_true, y_pred)
    print(f"Grammar QWK: {score:.4f}")
    return score

# === Load model regresi ===
def load_regression_model(path="models/model_logistic_regression.pkl"):
    return joblib.load(path)

# === Prediksi untuk 1 esai ===
def predict_grammar_score(features, model):
    features_df = pd.DataFrame([features])  # Convert dict to DataFrame
    predicted_score = model.predict(features_df)[0]
    return predicted_score

# === Prediksi untuk banyak esai dari CSV ===
def predict_grammar_scores_from_file(input_csv, output_csv, model):
    df = pd.read_csv(input_csv)

    scores, true_scores = [], []

    for i, essay in enumerate(df['essay']):
        grammar_features, _ = extract_grammar_features(essay)
        score = predict_grammar_score(grammar_features, model)
        scores.append(score)

        # Ambil ground truth jika tersedia
        if 'true_score' in df.columns:
            true_scores.append(df.loc[i, 'true_score'])

        print(f"Essay {i+1}/{len(df)} processed.")

    df['grammar_score'] = scores
    df.to_csv(output_csv, index=False)
    print(f"Skor grammar disimpan di {output_csv}")

    # Hitung QWK jika tersedia ground truth
    if true_scores:
        evaluate_qwk(true_scores, scores)