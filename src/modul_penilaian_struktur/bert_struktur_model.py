import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
import json
import logging
from sklearn.metrics import cohen_kappa_score
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

# === Logging Setup ===
log_file_path = 'logs/training/modul_penilaian_struktur/log_penilaian_struktur.txt'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# === Gunakan GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Menggunakan perangkat: {device}")

# === Load Dataset ===
train_df = pd.read_csv('data/pre-processing-data/training/training_data.csv')
val_df = pd.read_csv('data/pre-processing-data/validation/validation_data.csv')

logger.info(f"Jumlah data TRAINING: {len(train_df)}")
logger.info(f"Jumlah data VALIDATION: {len(val_df)}")

# === Tokenizer ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_dataset(df, label_column='skor_struktur_normalized'):
    inputs = tokenizer(df['essay'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    labels = torch.tensor(df[label_column].values, dtype=torch.float)
    return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

train_dataset = tokenize_dataset(train_df)
val_dataset = tokenize_dataset(val_df)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Model ===
class StructureScoreModel(nn.Module):
    def __init__(self):
        super(StructureScoreModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm1 = nn.LSTM(768, 400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(800, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x, _ = self.lstm1(bert_out.last_hidden_state)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return self.relu(x)

model = StructureScoreModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# === Training ===
logger.info("Mulai training model struktur esai...")
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start = time.time()

    for batch in train_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = criterion(preds.flatten(), labels.flatten())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    end = time.time()
    logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_dataloader):.4f} - Waktu: {end-start:.2f} detik")

# === Evaluasi QWK ===
def evaluate_qwk(model, dataloader):
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            preds = model(input_ids, attention_mask)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(np.clip(np.round(preds.cpu().numpy().reshape(-1)), 0, 10).astype(int))
    true_labels = np.round(true_labels).astype(int)
    qwk = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')
    return qwk, pred_labels, true_labels

train_qwk, train_preds, train_trues = evaluate_qwk(model, DataLoader(train_dataset, batch_size=16))
val_qwk, val_preds, val_trues = evaluate_qwk(model, val_dataloader)

logger.info(f"QWK - TRAINING:   {train_qwk:.4f}")
logger.info(f"QWK - VALIDATION: {val_qwk:.4f}")

# === Simpan Model & Prediksi ===
save_path = "models/bert_struktur_model"
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
torch.save(model.state_dict(), os.path.join(save_path, "structure_model.bin"))
with open(os.path.join(save_path, "config.json"), "w") as f:
    json.dump({
        "bert_model": "bert-base-uncased",
        "lstm1_hidden_size": 400,
        "lstm2_hidden_size": 128,
        "dropout": 0.5,
        "output_dim": 1,
        "max_length": 512
    }, f, indent=4)

results_dir = "logs/training/modul_penilaian_struktur"
os.makedirs(results_dir, exist_ok=True)

pd.DataFrame({
    'essay': train_df['essay'],
    'true_score': train_trues,
    'predicted_score': train_preds
}).to_csv(os.path.join(results_dir, 'train_predictions.csv'), index=False)

pd.DataFrame({
    'essay': val_df['essay'],
    'true_score': val_trues,
    'predicted_score': val_preds
}).to_csv(os.path.join(results_dir, 'val_predictions.csv'), index=False)

logger.info("Model dan hasil prediksi berhasil disimpan.")