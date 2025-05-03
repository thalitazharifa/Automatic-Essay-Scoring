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
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
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
        output_dir="./models/bert_cola_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=1.5e-5,
        weight_decay=0.01,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=300,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        logging_dir="./logs",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Melatih model grammar checker...")
    trainer.train()
    print("Model grammar checker telah dilatih.")
    return trainer

# === 6. Simpan Model dan Konversi ke ONNX ===
def save_model(model, tokenizer, save_path="./models/bert_cola_model_v2"):
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
    train_file = "E:/Bebeb/NEW/AES/data/cola_public/clean_train.csv"
    val_file = "E:/Bebeb/NEW/AES/data/cola_public/clean_val.csv"

    train_dataset, val_dataset, tokenizer = prepare_dataset(train_file, val_file)
    model = load_model()
    trainer = train_model(model, train_dataset, val_dataset)
    save_model(trainer.model, tokenizer)