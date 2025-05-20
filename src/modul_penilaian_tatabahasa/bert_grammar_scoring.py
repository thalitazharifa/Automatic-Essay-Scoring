import os
import sys
import joblib
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# === Setup Path ke Modul Lokal ===
sys.path.append(os.path.abspath("AES/src"))

# === Import Fungsi Eksternal ===
from modul_pengecekan_tatabahasa.main import load_grammar_checker_model
from modul_pengecekan_tatabahasa.fitur_tata_bahasa import extract_grammar_features

# === Setup Logging ===
log_dir = 'logs/training/modul_penilaian_tata_bahasa'
os.makedirs(log_dir, exist_ok=True)

log_grammar_path = os.path.join(log_dir, 'grammar_checker.log')
log_regression_path = os.path.join(log_dir, 'logistic_regression.log')

# Hapus handler logging sebelumnya
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Stream handler untuk log ke console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    return logger

grammar_logger = setup_logger("grammar_logger", log_grammar_path)
regression_logger = setup_logger("regression_logger", log_regression_path)

# === Tahap 1: Load dan Validasi Data ===
regression_logger.info("Memuat data esai dari CSV...")
df = pd.read_csv("data/pre-processing-data/training/training_data.csv")
essays = df["essay"]
scores = df["skor_tata_bahasa_normalized"]
regression_logger.info(f"Total esai dimuat: {len(df)}")

if essays.isnull().any():
    regression_logger.warning("Terdapat esai kosong dalam dataset.")
if scores.isnull().any():
    regression_logger.warning("Terdapat skor grammar kosong dalam dataset.")

# === Tahap 2: Load Grammar Checker Model (BERT) ===
grammar_logger.info("Memuat grammar checker model berbasis BERT...")
try:
    tokenizer, model, device = load_grammar_checker_model()
    grammar_logger.info("Model grammar checker berhasil dimuat.")
except Exception as e:
    grammar_logger.error(f"Gagal memuat model grammar: {e}")
    sys.exit(1)

# === Tahap 3: Ekstraksi Fitur Grammar dari Setiap Esai ===
grammar_logger.info("Memulai ekstraksi fitur grammar...")
feature_list = []

# Ekstraksi fitur untuk setiap esai
for i, essay in enumerate(essays):
    try:
        grammar_features, _ = extract_grammar_features(essay)
        feature_list.append(grammar_features)
    except Exception as e:
        grammar_logger.warning(f"Terjadi kesalahan pada esai {i+1}: {e}")
        feature_list.append([None] * 13)
    if (i + 1) % 100 == 0 or i == len(essays) - 1:
        grammar_logger.info(f"  - {i + 1}/{len(essays)} esai diproses")

# Simpan hasil ekstraksi fitur ke DataFrame
feature_df = pd.DataFrame(feature_list)
feature_df["score"] = scores
grammar_logger.info("Ekstraksi fitur grammar selesai.")

# === Tahap 4: Pelatihan dan Evaluasi Model Regresi ===
regression_logger.info("Menyiapkan data pelatihan regresi...")
X = feature_df.drop(columns=["score"]) # Fitur grammar
y = feature_df["score"] # Skor grammar target (label)

# Pelatihan model regresi logistik
regression_logger.info("Melatih model regresi logistik...")
model_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model_lr.fit(X, y)
regression_logger.info("Pelatihan model selesai.")

# Mengevaluasi model dengan Quadratic Weighted Kappa
regression_logger.info("Mengevaluasi model dengan Quadratic Weighted Kappa...")
y_pred = model_lr.predict(X)
qwk = cohen_kappa_score(y, y_pred, weights='quadratic')
regression_logger.info(f"QWK (Training - Full Data): {qwk:.4f}")

# Simpan hasil evaluasi
eval_result_path = os.path.join(log_dir, "evaluasi_model_training.txt")
with open(eval_result_path, "w") as f:
    f.write("Hasil Evaluasi Model Penilaian Tata Bahasa (Training)\n")
    f.write("===============================================\n")
    f.write(f"Jumlah data latih: {len(X)}\n")
    f.write(f"Fitur grammar: {list(X.columns)}\n")
    f.write(f"QWK (Training): {qwk:.4f}\n")
regression_logger.info(f"Hasil evaluasi ditulis ke: {eval_result_path}")

# === Tahap 5: Simpan Model dan Prediksi ===
model_path = os.path.join(log_dir, "model_logistic_regression.pkl")
output_csv_path = os.path.join(log_dir, "hasil_prediksi_grammar_training.csv")

# Simpan model dan hasil prediksi
joblib.dump(model_lr, model_path)
regression_logger.info(f"Model regresi disimpan di: {model_path}")

# Simpan hasil prediksi ke file CSV
feature_df["predicted_score"] = model_lr.predict(X)
feature_df.to_csv(output_csv_path, index=False)
regression_logger.info(f"Hasil prediksi disimpan di: {output_csv_path}")

regression_logger.info("Proses penilaian grammar selesai dengan sukses.")

# === Tahap 6: Memuat Data Validasi ===
regression_logger.info("Memuat data validasi dari CSV...")
df_validation = pd.read_csv("data/pre-processing-data/validation/validation_data.csv")
essays_validation = df_validation["essay"]
scores_validation = df_validation["skor_tata_bahasa_normalized"]
regression_logger.info(f"Total esai validasi dimuat: {len(df_validation)}")

if essays_validation.isnull().any():
    regression_logger.warning("Terdapat esai kosong dalam dataset validasi.")
if scores_validation.isnull().any():
    regression_logger.warning("Terdapat skor grammar kosong dalam dataset validasi.")

# === Tahap 7: Ekstraksi Fitur Grammar dari Setiap Esai Validasi ===
grammar_logger.info("Memulai ekstraksi fitur grammar dari data validasi...")
feature_list_validation = []

for i, essay in enumerate(essays_validation):
    try:
        grammar_features, _ = extract_grammar_features(essay)
        feature_list_validation.append(grammar_features)
    except Exception as e:
        grammar_logger.warning(f"Terjadi kesalahan pada esai validasi {i+1}: {e}")
        feature_list_validation.append([None] * 13) 
    
    if (i + 1) % 100 == 0 or i == len(essays_validation) - 1:
        grammar_logger.info(f"  - {i + 1}/{len(essays_validation)} esai validasi diproses")

feature_df_validation = pd.DataFrame(feature_list_validation)
feature_df_validation["score"] = scores_validation
grammar_logger.info("Ekstraksi fitur grammar dari data validasi selesai.")

# === Tahap 8: Evaluasi Model pada Data Validasi ===
regression_logger.info("Menyiapkan data untuk evaluasi model pada data validasi...")
X_validation = feature_df_validation.drop(columns=["score"])
y_validation = feature_df_validation["score"]

# Evaluasi model pada data validasi
regression_logger.info("Mengevaluasi model dengan data validasi...")
y_pred_validation = model_lr.predict(X_validation)
qwk_validation = cohen_kappa_score(y_validation, y_pred_validation, weights='quadratic')
regression_logger.info(f"QWK (Validation): {qwk_validation:.4f}")

# Simpan hasil evaluasi validasi ke file
eval_result_validation_path = os.path.join(log_dir, "evaluasi_model_validation.txt")
with open(eval_result_validation_path, "w") as f:
    f.write("Hasil Evaluasi Model Penilaian Tata Bahasa (Validasi)\n")
    f.write("=====================================================\n")
    f.write(f"Jumlah data validasi: {len(X_validation)}\n")
    f.write(f"Fitur grammar: {list(X_validation.columns)}\n")
    f.write(f"QWK (Validation): {qwk_validation:.4f}\n")
regression_logger.info(f"Hasil evaluasi validasi ditulis ke: {eval_result_validation_path}")

# === Tahap 9: Simpan Prediksi pada Data Validasi ===
output_validation_csv_path = os.path.join(log_dir, "hasil_prediksi_grammar_validation.csv")

# Simpan hasil prediksi validasi ke file CSV
feature_df_validation["predicted_score"] = y_pred_validation
feature_df_validation.to_csv(output_validation_csv_path, index=False)
regression_logger.info(f"Hasil prediksi validasi disimpan di: {output_validation_csv_path}")

regression_logger.info("Proses evaluasi dan prediksi validasi selesai dengan sukses.")