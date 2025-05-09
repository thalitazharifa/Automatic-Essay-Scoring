import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
import json
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score

# === Gunakan GPU jika tersedia ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

# === Load Dataset ===
data = pd.read_csv('E:\Kuliah\Tugas Akhir\AES\code\AES\data\pre-processing-data\training\training_data.csv')

# === Tokenisasi ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
essays = data['essay'].tolist()
inputs = tokenizer(essays, return_tensors='pt', padding=True, truncation=True, max_length=512)

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = torch.tensor(data['skor_struktur_normalized'].values, dtype=torch.float)

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# === Definisi Model ===
class StructureScoreModel(nn.Module):
    def __init__(self):
        super(StructureScoreModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Aktifkan fine-tuning BERT
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm1 = nn.LSTM(768, 400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(800, 128, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out.last_hidden_state  # (batch_size, seq_len, 768)

        # Ambil token [CLS] (posisi 0) → representasi global esai
        cls_token = hidden_states[:, 0, :].unsqueeze(1)  # (batch_size, 1, 768)

        x, _ = self.lstm1(cls_token)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]  # ambil token terakhir
        x = self.fc(x)
        return self.relu(x)

# === Inisialisasi model, loss, dan optimizer ===
model = StructureScoreModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# === Training ===
num_epochs = 50
log_file = "training_log.txt"
with open(log_file, "w") as f:
    f.write("Epoch, Loss, Waktu (detik)\n")

for epoch in range(num_epochs):
    start = time.time()
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]

        optimizer.zero_grad()
        preds = model(input_ids_batch, attention_mask_batch)
        loss = criterion(preds.flatten(), labels_batch.flatten())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    end = time.time()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {end-start:.2f} detik")
    with open(log_file, "a") as f:
        f.write(f"{epoch+1}, {avg_loss:.4f}, {end-start:.2f}\n")

print("Training selesai!")

# === Save Model, Tokenizer, and Config ===
save_path = "models/bert_struktur_model"
os.makedirs(save_path, exist_ok=True)

# 1. Save tokenizer
tokenizer.save_pretrained(save_path)

# 2. Save custom config
custom_config = {
    "bert_model": "bert-base-uncased",
    "lstm1_hidden_size": 400,
    "lstm2_hidden_size": 128,
    "dropout": 0.5,
    "output_dim": 1,
    "max_length": 512
}
with open(os.path.join(save_path, "config.json"), "w") as f:
    json.dump(custom_config, f, indent=4)

# 3. Save model weights
torch.save(model.state_dict(), os.path.join(save_path, "structure_model.bin"))

# 4. Save ONNX
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 512)).to(device)
dummy_attention_mask = torch.ones((1, 512)).to(device)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    os.path.join(save_path, "structure_model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["structure_score"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "structure_score": {0: "batch_size"}
    }
)

print(f"Model lengkap disimpan di: {save_path}")

    # === Evaluasi (QWK) ===
def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=10):
    y_true = np.clip(np.round(y_true), min_rating, max_rating).astype(int)
    y_pred = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

model.eval()
true_labels, pred_labels = [], []

results_save_folder = "E:\Kuliah\Tugas Akhir\AES\code\AES\data\result\modul_penilaian_struktur"
os.makedirs(results_save_folder, exist_ok=True)

with torch.no_grad():
    for batch in dataloader:
        input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
        preds = model(input_ids_batch, attention_mask_batch)
        true_labels.extend(labels_batch.cpu().numpy())
        preds_np = preds.detach().cpu().numpy().reshape(-1)
        rounded_preds = np.clip(np.round(preds_np), 0, 10).astype(int)
        pred_labels.extend(rounded_preds.tolist())

qwk = quadratic_weighted_kappa(np.array(true_labels), np.array(pred_labels))
print(f"QWK di data training: {qwk:.4f}")

with open("qwk_training_log.txt", "w") as f:
    f.write(f"QWK di data training: {qwk:.4f}\n")

results = pd.DataFrame({
    'essay': essays,
    'true_score': true_labels,
    'predicted_score': pred_labels
})

results.to_csv(f"{results_save_folder}/modul_penilaian_struktur_qwk_training.csv", index=False)
print(f"Hasil prediksi disimpan di: {results_save_folder}")