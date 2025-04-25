# Import library
import pandas as pd

# Membaca dataset dari file .tsv
raw_data = pd.read_csv('E:/Kuliah/Tugas Akhir/AES/code/AES/data/asap-aes/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')

# Menyaring data untuk hanya mengambil essay_set dengan nilai 7
filtered_dataset = raw_data[raw_data['essay_set'] == 7].copy()

# Menyimpan dataset yang telah disaring ke dalam file CSV
filtered_dataset.to_csv('aes_dataset_value_7.csv', index=False)

# Menentukan kolom yang akan digabungkan untuk skor Struktur
skor_struktur = ['rater1_trait1', 'rater1_trait2', 'rater1_trait3', 'rater2_trait1', 'rater2_trait2', 'rater2_trait3']

# Menghitung skor struktur sebagai rata-rata dari 6 karakteristik yang ada
filtered_dataset.loc[:, 'skor_struktur'] = filtered_dataset[skor_struktur].sum(axis=1) / len(skor_struktur)

# Menentukan kolom yang akan digabungkan untuk skor Tata Bahasa
skor_tata_bahasa = ['rater1_trait4', 'rater2_trait4']

# Menghitung skor tata bahasa sebagai rata-rata dari 2 karakteristik yang ada
filtered_dataset.loc[:, 'skor_tata_bahasa'] = filtered_dataset[skor_tata_bahasa].sum(axis=1) / len(skor_tata_bahasa)

# Normalisasi skor Struktur dalam rentang 0-10 (bulat)
min_value_struktur = filtered_dataset['skor_struktur'].min()
max_value_struktur = filtered_dataset['skor_struktur'].max()
filtered_dataset['skor_struktur_normalized'] = (
    10 * (filtered_dataset['skor_struktur'] - min_value_struktur) / 
    (max_value_struktur - min_value_struktur)
).round(0).astype(int)

# Normalisasi skor Tata Bahasa dalam rentang 0-10 (bulat)
min_value_tata_bahasa = filtered_dataset['skor_tata_bahasa'].min()
max_value_tata_bahasa = filtered_dataset['skor_tata_bahasa'].max()
filtered_dataset['skor_tata_bahasa_normalized'] = (
    10 * (filtered_dataset['skor_tata_bahasa'] - min_value_tata_bahasa) / 
    (max_value_tata_bahasa - min_value_tata_bahasa)
).round(0).astype(int)

# Pilih kolom-kolom yang relevan untuk disimpan dalam dataset hasil pre-processing
pre_processing_data = ['essay_id', 'essay_set', 'essay', 'skor_struktur_normalized', 'skor_tata_bahasa_normalized']

# Membuat dataset akhir yang berisi data yang sudah diproses
final_pre_processing_data = filtered_dataset[pre_processing_data]

# Menyimpan data hasil pre-processing ke dalam file CSV
final_pre_processing_data.to_csv('data/result/pre_processing_data.csv', index=False)

print("Proses pre-processing selesai, file telah disimpan sebagai 'pre_processing_data.csv'")