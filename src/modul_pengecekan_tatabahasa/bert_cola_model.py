import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# === 1. Gunakan GPU jika tersedia ===
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Menggunakan perangkat: {device}")

# === 2. Load dan Persiapkan Dataset ===
def load_clean_data(file_path):
    df = pd.read_csv(file_path)
    return df[['sentence', 'label']]

def prepare_dataset(train_file, val_file):
    train_df = load_clean_data(train_file)
    val_df = load_clean_data(val_file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Atau sesuaikan dengan model lain

    def tokenize(df):
        encodings = tokenizer(
            list(df["sentence"]),
            padding="max_length",
            truncation=True,
            max_length=128
        )
        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label": df["label"].tolist()
        })

    train_dataset = tokenize(train_df)
    val_dataset = tokenize(val_df)
    return train_dataset, val_dataset, tokenizer

# === 3. Load dan Konfigurasi Model BERT ===
def load_model():
    # Menyesuaikan model yang digunakan
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # bisa ganti ke model yang sesuai
    model.to(device)
    return model

# === 4. Definisi Evaluasi Model ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary")
    }

# === 5. Fungsi Pelatihan ===
def train_model(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./models/bert_cola_model",  # Output model
        evaluation_strategy="epoch",  # Evaluasi setiap epoch
        save_strategy="epoch",  # Simpan model setiap epoch
        save_total_limit=2,  # Hanya simpan dua model terbaik
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Pilih metrik terbaik model berdasarkan eval_loss
        learning_rate=1e-5,  # Menyesuaikan learning rate
        weight_decay=0.01,
        num_train_epochs=4,  # Jumlah epoch yang lebih sedikit untuk eksperimen cepat
        per_device_train_batch_size=16,  # Ukuran batch lebih besar untuk training
        per_device_eval_batch_size=16,  # Ukuran batch lebih besar untuk evaluasi
        warmup_steps=500,  # Jumlah warmup steps untuk stabilisasi
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        logging_dir="./logs",  # Tempat untuk log
        logging_steps=10,  # Interval logging
        report_to=[],  # Jangan kirim log ke platform lain
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Penghentian awal jika tidak ada kemajuan
    )

    print("Melatih model grammar checker...")
    trainer.train()
    print("Model grammar checker telah dilatih.")
    return trainer

# === 6. Simpan Model dan Konversi ke ONNX ===
def save_model(model, tokenizer, save_path="./models/bert_cola_model"):
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128)).to(device)
    dummy_attention_mask = torch.ones((1, 128)).to(device)

    model.eval()
    torch.onnx.export(
        model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=os.path.join(save_path, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=16
    )

    print(f"Model grammar checker disimpan di: {save_path}")
    print(f"Model ONNX tersedia di: {save_path}/model.onnx")

# === 7. Jalankan Seluruh Pipeline ===
if __name__ == "__main__":
    train_file = "E:\Kuliah\Tugas Akhir\AES\code\AES\data\cola_public\raw\in_domain_train.csv"  # raw/in_domain_train.csv
    val_file = "E:\Kuliah\Tugas Akhir\AES\code\AES\data\cola_public\raw\in_domain_dev.csv"      # raw/in_domain_dev.csv

    train_dataset, val_dataset, tokenizer = prepare_dataset(train_file, val_file)
    model = load_model()
    trainer = train_model(model, train_dataset, val_dataset)

    # === Evaluasi Model ===
    print("\nEvaluasi model pada validation set:")
    eval_result = trainer.evaluate()
    print(f"Akurasi: {eval_result['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_result['eval_f1']:.4f}")
    print(f"Eval Loss: {eval_result['eval_loss']:.4f}")

    # === Simpan Model ===
    save_model(trainer.model, tokenizer)