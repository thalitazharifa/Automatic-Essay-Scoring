import torch
import torch.nn as nn
import pandas as pd
import time
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import os

# === 1. Gunakan GPU jika tersedia ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

# === 2. Load Dataset ===
def load_dataset(file_path):
    return pd.read_csv(file_path)

# === 3. Tokenisasi Esai ===
def tokenize_essays(essay_texts, tokenizer):
    return tokenizer(essay_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# === 4. Definisi Model Struktur Esai ===
class StructureScoreModel(nn.Module):
    def __init__(self, hidden_size=768):
        super(StructureScoreModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=800, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(128 * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]  # Ambil output terakhir LSTM
        x = self.dense(x)
        x = self.relu(x)
        return x

# === 5. Load Model BERT dan Tokenizer ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# === 6. Load Data & Preprocessing ===
data_preprocessing = load_dataset('E:/Kuliah/Tugas Akhir/AES/code/AES/data/pre-processing-data/training/training_data.csv')
essay_texts = data_preprocessing['essay'].tolist()
tokenized_inputs = tokenize_essays(essay_texts, tokenizer)

input_ids = tokenized_inputs['input_ids'].to(device)
attention_mask = tokenized_inputs['attention_mask'].to(device)

# Skor Struktur Esai
structure_scores = torch.tensor(data_preprocessing['skor_struktur_normalized'].values, dtype=torch.float).to(device)

# === 7. Dataset & DataLoader ===
train_dataset = TensorDataset(input_ids, attention_mask, structure_scores)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# === 8. Inisialisasi Model Struktur ===
model_structure = StructureScoreModel(hidden_size=768).to(device)

# === 9. Loss Function & Optimizer ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_structure.parameters(), lr=1e-5)

# === 10. Training Loop ===
num_epochs = 50
log_file = "training_log.txt"
with open(log_file, "w") as f:
    f.write("Epoch, Loss, Waktu (detik)\n")

for epoch in range(num_epochs):
    start_time = time.time()
    model_structure.train()
    epoch_loss = 0

    for batch in train_dataloader:
        input_ids_batch, attention_mask_batch, structure_scores_batch = batch
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            outputs = bert_model(input_ids_batch, attention_mask=attention_mask_batch)
        bert_embeddings = outputs.last_hidden_state
        predictions = model_structure(bert_embeddings)

        loss = criterion(predictions.squeeze(), structure_scores_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Waktu: {elapsed_time:.2f} detik")
    
    with open(log_file, "a") as f:
        f.write(f"{epoch+1}, {avg_loss:.4f}, {elapsed_time:.2f}\n")

print("Training complete!")

# === 11. Simpan Model ===
def save_model(model_structure, tokenizer, save_path="models/bert_struktur_model"):
    os.makedirs(save_path, exist_ok=True)

    # Simpan model PyTorch
    torch.save(model_structure.state_dict(), f"{save_path}/structure_model.bin")

    # Simpan model ONNX
    dummy_input = torch.randn(1, 512, 768).to(device)
    torch.onnx.export(
        model_structure,
        dummy_input,
        f"{save_path}/structure_model.onnx",
        input_names=["bert_embeddings"],
        output_names=["structure_score"],
        dynamic_axes={"bert_embeddings": {0: "batch_size"}, "structure_score": {0: "batch_size"}}
    )

    print(f"Model telah disimpan di: {save_path}")
    print(f"Model universal tersedia dalam format ONNX: {save_path}/structure_model.onnx")

save_model(model_structure, tokenizer)
print(f"Log pelatihan disimpan dalam {log_file}")