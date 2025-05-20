import pandas as pd
import numpy as np
import os

# Membaca dataset dari file .tsv
raw_data = pd.read_csv('data\asap-aes\training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')

# Menyaring data untuk hanya mengambil essay_set dengan nilai 7
filtered_dataset = raw_data[raw_data['essay_set'] == 7].copy()

# Menyimpan dataset yang telah disaring ke dalam file CSV
output_file = 'data/pre-processing-data/aes_dataset_value_7.csv'
filtered_dataset.to_csv(output_file, index=False)

# Menentukan kolom yang akan digabungkan untuk skor Struktur
skor_struktur = ['rater1_trait1', 'rater1_trait2', 'rater1_trait3', 'rater2_trait1', 'rater2_trait2', 'rater2_trait3']

# Menghitung skor struktur sebagai rata-rata dari 6 karakteristik yang ada
filtered_dataset.loc[:, 'skor_struktur'] = filtered_dataset[skor_struktur].sum(axis=1) / len(skor_struktur)

# Menentukan kolom yang akan digabungkan untuk skor Tata Bahasa
skor_tata_bahasa = ['rater1_trait4', 'rater2_trait4']

# Menghitung skor tata bahasa sebagai rata-rata dari 2 karakteristik yang ada
filtered_dataset.loc[:, 'skor_tata_bahasa'] = filtered_dataset[skor_tata_bahasa].sum(axis=1) / len(skor_tata_bahasa)

# Fungsi custom rounding: 0.5 ke atas, di bawah 0.5 ke bawah
def custom_round(series):
    return np.where(series - np.floor(series) < 0.5, np.floor(series), np.ceil(series)).astype(int)

# Normalisasi skor Struktur dalam rentang 0-10 (bulat)
min_value_struktur = filtered_dataset['skor_struktur'].min()
max_value_struktur = filtered_dataset['skor_struktur'].max()
filtered_dataset['skor_struktur_normalized'] = custom_round(
    10 * (filtered_dataset['skor_struktur'] - min_value_struktur) / (max_value_struktur - min_value_struktur)
)

# Normalisasi skor Tata Bahasa dalam rentang 0-10 (bulat)
min_value_tata_bahasa = filtered_dataset['skor_tata_bahasa'].min()
max_value_tata_bahasa = filtered_dataset['skor_tata_bahasa'].max()
filtered_dataset['skor_tata_bahasa_normalized'] = custom_round(
    10 * (filtered_dataset['skor_tata_bahasa'] - min_value_tata_bahasa) / (max_value_tata_bahasa - min_value_tata_bahasa)
)

# Pilih kolom-kolom yang relevan untuk disimpan dalam dataset hasil pre-processing
pre_processing_data = ['essay_id', 'essay_set', 'essay', 'skor_struktur_normalized', 'skor_tata_bahasa_normalized']

# Membuat dataset akhir yang berisi data yang sudah diproses
final_pre_processing_data = filtered_dataset[pre_processing_data]

# Tentukan path untuk menyimpan file CSV
output_dir = 'data/pre-processing-data'
os.makedirs(output_dir, exist_ok=True)

# Menyimpan data hasil pre-processing ke dalam file CSV
pre_processed_file = os.path.join(output_dir, 'pre_processing_data.csv')
final_pre_processing_data.to_csv(pre_processed_file, index=False)

print(f"Proses pre-processing selesai, file telah disimpan sebagai '{pre_processed_file}'")