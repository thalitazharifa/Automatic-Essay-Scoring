import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer, BertModel
from sklearn.metrics import cohen_kappa_score

# Definisikan model
class StructureScoreModel(nn.Module):
    def __init__(self):
        super(StructureScoreModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = True  # Fine-tune BERT

        self.lstm1 = nn.LSTM(768, 400, batch_first=True, bidirectional=True)  # output: 800
        self.lstm2 = nn.LSTM(800, 128, batch_first=True, bidirectional=True)  # output: 256

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = bert_out.last_hidden_state  # shape: (batch_size, seq_len, 768)

        # Feed entire sequence to LSTM, not just [CLS] token
        x, _ = self.lstm1(hidden_states)  # shape: (batch_size, seq_len, 800)
        x, _ = self.lstm2(x)              # shape: (batch_size, seq_len, 256)

        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return self.relu(x)

# Fungsi untuk memuat model dan tokenizer
def load_structure_model(model_path="models/bert_struktur_model"):
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = StructureScoreModel()
    model.load_state_dict(torch.load(os.path.join(model_path, 'structure_model.bin')))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# Fungsi untuk menghitung Quadratic Weighted Kappa (QWK)
def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=10):
    y_true = np.clip(np.round(y_true), min_rating, max_rating).astype(int)
    y_pred = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Fungsi untuk melakukan prediksi skor struktur
def predict_structure_score(essays, model, tokenizer, device, batch_size=8):
    model.eval()
    predictions = []

    for i in range(0, len(essays), batch_size):
        batch_essays = essays[i:i+batch_size]
        inputs = tokenizer(batch_essays, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            preds = outputs.squeeze(-1).cpu().numpy()  # shape: (batch_size,)
            predictions.extend(preds)

    return np.array(predictions)

# Fungsi untuk menyimpan hasil prediksi ke file CSV
def save_predictions_to_csv(predictions, essays, filename="data/validation-penilaian-struktur.csv"):
    rounded_predictions = np.round(predictions + 0.5).astype(int)  # Tambahkan ini
    df = pd.DataFrame({
        'essay': essays,
        'predicted_structure_score': rounded_predictions  # Gunakan yang sudah dibulatkan
    })
    df.to_csv(filename, index=False)
    print(f"Predictions saved to: {filename}")

# Fungsi untuk menghitung QWK pada data
def evaluate_qwk(true_labels, pred_labels):
    qwk = quadratic_weighted_kappa(np.array(true_labels), np.array(pred_labels))
    print(f"QWK Score: {qwk:.4f}")
    return qwk