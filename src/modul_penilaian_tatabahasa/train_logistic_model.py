import pandas as pd
import joblib
import os
from modul_pengecekan_tatabahasa.main import load_grammar_checker_model
from modul_penilaian_tatabahasa.fitur_tata_bahasa import extract_grammar_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# === 1. Load Data Essay (Hanya essay_set 7) ===
df = pd.read_csv("E:\Kuliah\Tugas Akhir\AES\code\AES\data\pre-processing-data\training\training_data.csv")  # asumsikan sudah dinormalisasi skor 0-10
essays = df["essay"]
scores = df["skor_tata_bahasa_normalized"]  # kolom skor grammar hasil normalisasi rater

# === 2. Load Grammar Checker Model Sekali Saja ===
tokenizer, model, device = load_grammar_checker_model()

# === 3. Ekstraksi Fitur Grammar ===
print("Mengekstraksi fitur grammar dari esai...")
feature_list = []
for i, essay in enumerate(essays):
    # Menggunakan extract_grammar_features yang sudah ada
    grammar_features, _ = extract_grammar_features(essay)
    
    # # Memilih hanya fitur yang diperlukan
    # selected_features = {
    #     "SWg_new": grammar_features["SWg_new"],
    #     "Grammar_Ratio_Sentence": grammar_features["Grammar_Ratio_Sentence"],
    #     "CWg_new": grammar_features["CWg_new"],
    #     "Grammar_Ratio_Clause": grammar_features["Grammar_Ratio_Clause"]
    # }
    
    feature_list.append(grammar_features)

    if (i + 1) % 100 == 0:
        print(f"  - {i + 1} esai diproses")

# Mengonversi list fitur yang terpilih menjadi DataFrame
feature_df = pd.DataFrame(feature_list)
feature_df["score"] = scores

# === 4. Gunakan Semua Data untuk Training ===
X = feature_df.drop(columns=["score"])
y = feature_df["score"]

# === 5. Latih Model Regresi Logistik dengan Scaling ===
print("Melatih model regresi logistik...")
model_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model_lr.fit(X, y)

# === 6. Evaluasi Model dengan QWK (Training pada seluruh data) ===
y_pred = model_lr.predict(X)

# Hitung QWK untuk seluruh data
qwk = cohen_kappa_score(y, y_pred, weights='quadratic')

# Cetak Evaluasi Model
print(f"\nEvaluasi Model:")
print(f"  - QWK (Training - Full Data): {qwk:.4f}")

# === 7. Simpan Model dan Fitur ===
output_dir = "E:\Kuliah\Tugas Akhir\AES\code\AES\models"
os.makedirs(output_dir, exist_ok=True)  # pastikan folder ada
model_path = os.path.join(output_dir, "model_logistic_regression.pkl")

# Simpan model
joblib.dump(model_lr, model_path)

# Menambahkan nilai prediksi ke dalam DataFrame
feature_df["predicted_score"] = model_lr.predict(X)  # Menambahkan kolom prediksi ke DataFrame

# Simpan hasil prediksi grammar dan nilai sebenarnya ke dalam file CSV
output_csv_path = os.path.join(output_dir, "hasil_prediksi_grammar.csv")
feature_df.to_csv(output_csv_path, index=False)

print(f"\nModel dan hasil prediksi grammar berhasil disimpan ke {output_csv_path}.")