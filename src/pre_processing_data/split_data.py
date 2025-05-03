# Import library tambahan
import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca dataset hasil pre-processing
final_pre_processing_data = pd.read_csv('data/result/pre_processing_data.csv')

# Membagi dataset menjadi 80% training dan 20% validation
train_data, val_data = train_test_split(final_pre_processing_data, test_size=0.2, random_state=42)

# Menyimpan dataset training ke file CSV
train_data.to_csv('data/pre-processing-data/training/training_data.csv', index=False)

# Menyimpan dataset validation ke file CSV
val_data.to_csv('data/pre-processing-data/validation/validation_data.csv', index=False)

print("Dataset berhasil dipisahkan: 'training_data.csv' (80%) dan 'validation_data.csv' (20%)")