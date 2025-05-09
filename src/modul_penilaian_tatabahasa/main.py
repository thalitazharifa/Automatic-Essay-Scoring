# grammar_scoring/main.py

import pandas as pd
import joblib
from fitur_tata_bahasa import extract_grammar_features

# Load model regresi
def load_regression_model(path="E:\Kuliah\Tugas Akhir\AES\code\AES\models\model_logistic_regression.pkl"):
    return joblib.load(path)

# Fungsi untuk menghitung skor grammar untuk 1 esai
def predict_grammar_score(essay, model):
    features = extract_grammar_features(essay)
    df = pd.DataFrame([features])
    predicted_score = model.predict(df)[0]
    return max(0, min(10, round(predicted_score, 2)))  # Normalisasi 0–10

# Untuk batch prediksi jika pakai CSV
def predict_grammar_scores_from_file(input_csv, output_csv, model):
    df = pd.read_csv(input_csv)
    df['grammar_score'] = df['essay'].apply(lambda x: predict_grammar_score(x, model))
    df.to_csv(output_csv, index=False)
    print(f"Skor grammar disimpan di {output_csv}")