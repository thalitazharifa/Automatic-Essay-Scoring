import os
import torch
import joblib
import streamlit as st
import pandas as pd
import sys
from transformers import BertTokenizerFast, BertForSequenceClassification

sys.path.append(os.path.abspath(r"E:\Kuliah\Tugas Akhir\AES\code\AES\src"))

from modul_pengecekan_tatabahasa.fitur_tata_bahasa import extract_grammar_features
from modul_penilaian_struktur.main import StructureScoringModel

st.set_page_config(page_title="Automated Essay Scoring", layout="centered")

# === Load Grammar Scoring Model ===
grammar_model_path = r"E:\Kuliah\Tugas Akhir\AES\code\AES\models\model_logistic_regression.pkl"
grammar_model = joblib.load(grammar_model_path)

# === Load Struktur Scoring Model Weights ===
struktur_model_path = r"E:\Kuliah\Tugas Akhir\AES\code\AES\models\bert_struktur_model\2e-5\bert_struktur.pth"
struktur_model_weights = torch.load(struktur_model_path, map_location=torch.device("cpu"))

# === Load Grammar Checker BERT Model ===
@st.cache_resource
def load_grammar_checker_model(model_path=r"E:\Kuliah\Tugas Akhir\AES\code\AES\models\bert_cola_model\2e-5"):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # type: ignore
    model.eval()
    return tokenizer, model, device

grammar_tokenizer, grammar_bert_model, grammar_device = load_grammar_checker_model()

# === Inisialisasi dan load model struktur ===
struktur_model = StructureScoringModel()
struktur_model.load_state_dict(struktur_model_weights, strict=False)
struktur_model.eval()

struktur_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# === Fungsi Prediksi Struktur Skor Detail ===
def predict_structure_scores_detail(essay):
    inputs = struktur_tokenizer(
        essay,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        ideas, org, style = struktur_model(inputs["input_ids"], inputs["attention_mask"])
        scores = {
            "ideas": round(ideas.item()),
            "organization": round(org.item()),
            "style": round(style.item()),
            "total": 2 * round(ideas.item()) + round(org.item()) + round(style.item())
        }
    return scores

# === Fungsi Prediksi Grammar Skor ===
def predict_grammar_score(essay):
    grammar_features, _ = extract_grammar_features(essay, grammar_tokenizer, grammar_bert_model, grammar_device)
    grammar_features_df = pd.DataFrame([grammar_features])
    predicted_score = grammar_model.predict(grammar_features_df)[0]
    return predicted_score

# === Streamlit UI ===
st.title("📝 Automated Essay Scoring (AES)")
st.markdown("Selamat datang di sistem penilaian esai otomatis! Masukkan esai Bahasa Inggris Anda di bawah untuk mendapatkan analisis menyeluruh terhadap **struktur** dan **tata bahasa**.")

with st.expander("ℹ️ Panduan Penilaian"):
    st.markdown("""
    **🔍 Kriteria Penilaian:**

    - **Konvensi / Grammar (0–3):**  
      Penggunaan konvensi Bahasa Inggris Baku yang mencakup tata bahasa, penggunaan, dan ejaan:  
        - **3:** Konsisten dan tepat  
        - **2:** Memadai  
        - **1:** Terbatas  
        - **0:** Tidak efektif  

    - **Struktur (0–12):** Terdiri dari tiga aspek berikut:  
        - **Gagasan / Ideas (×2 bobot)**  
            - **3:** Fokus jelas, berkembang sepenuhnya dengan detail spesifik dan relevan  
            - **2:** Cukup fokus, berkembang dengan campuran detail spesifik/umum  
            - **1:** Fokus lemah, berkembang dengan detail terbatas/umum  
            - **0:** Tidak fokus dan/atau tidak berkembang  
        - **Organisasi / Organization**  
            - **3:** Hubungan antar gagasan/peristiwa jelas dan logis  
            - **2:** Susunan gagasan/peristiwa logis  
            - **1:** Hubungan gagasan/peristiwa lemah  
            - **0:** Tidak ada organisasi  
        - **Gaya Penulisan / Style**  
            - **3:** Pilihan kata menarik, variasi struktur kalimat mendukung tujuan dan audiens  
            - **2:** Bahasa cukup efektif dan jelas  
            - **1:** Pilihan kata/kalimat terbatas, menghambat komunikasi  
            - **0:** Bahasa tidak mendukung tujuan dan audiens  

    - **Skor Holistik (0–15):**  
      Merupakan penjumlahan dari skor grammar dan struktur.  
      - Struktur = (Ideas × 2) + Organization + Style
      - Holistik = (Ideas × 2) + Organization + Style + Grammar
               = Struktur + Grammar
    """)

essay_input = st.text_area("✍️ Tulis esai Anda di sini:", height=300, placeholder="Contoh: In today's world, technology plays a crucial role...")

if st.button("✅ Nilai Esai"):
    if essay_input.strip() == "":
        st.warning("⚠️ Mohon masukkan teks esai terlebih dahulu.")
    else:
        with st.spinner("🔍 Sistem sedang menilai esai Anda..."):
            grammar_score = predict_grammar_score(essay_input)
            struktur_scores = predict_structure_scores_detail(essay_input)
            holistik_score = grammar_score + struktur_scores["total"]

        st.success("🎉 Penilaian selesai!")

        st.markdown("---")
        st.subheader("📊 Hasil Penilaian")
        col1, col2, col3 = st.columns(3)
        col1.metric("Grammar", f"{grammar_score}/3")
        col2.metric("Structure", f"{struktur_scores['total']}/12")
        col3.metric("Holistik", f"{holistik_score}/15")

        with st.expander("🔍 Rincian Penilaian Struktur"):
            st.markdown(f"- **Ideas (×2):** {struktur_scores['ideas']} → {struktur_scores['ideas']*2}")
            st.markdown(f"- **Organization:** {struktur_scores['organization']}")
            st.markdown(f"- **Style:** {struktur_scores['style']}")
