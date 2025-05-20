import os
import torch
import pandas as pd
import numpy as np
import logging
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import time

# === Logging Setup ===
log_dir = 'logs/training/modul_pengecekan_tata_bahasa'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'log_training_pengecekan_tata_bahasa.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

# === 1. Gunakan GPU jika tersedia ===
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
logger.info(f"Menggunakan perangkat: {device}")

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
            max_length=512
        )
        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label": df["label"].tolist()
        })

    train_dataset = tokenize(train_df)
    val_dataset = tokenize(val_df)
    logger.info("Dataset training dan validation berhasil dipersiapkan.")
    return train_dataset, val_dataset, tokenizer

# === 3. Load dan Konfigurasi Model BERT ===
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    logger.info("Model BERT berhasil dimuat dan dikirim ke perangkat.")
    return model

# === 4. Definisi Evaluasi Model ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)
    
    logger.info(f"Confusion Matrix: \n{cm}")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# === 5. Fungsi Pelatihan ===
def train_model(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./models/bert_cola_model",
        evaluation_strategy="epoch",  # Evaluasi setiap epoch
        save_strategy="epoch",  # Simpan model setiap epoch
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        gradient_accumulation_steps=1,
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

    logger.info("Mulai proses pelatihan model grammar checker...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info(f"Pelatihan selesai. Total waktu pelatihan: {total_train_time:.2f} detik.")
    return trainer

# === 6. Simpan Model ===
def save_model(model, tokenizer, save_path="./models/bert_cola_model"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    logger.info(f"Model grammar checker disimpan di: {save_path}")

# === 7. Jalankan Seluruh Pipeline ===
if __name__ == "__main__":
    train_file = "data/cola_public/raw/in_domain_train.csv"
    val_file = "data/cola_public/raw/in_domain_dev.csv"

    train_dataset, val_dataset, tokenizer = prepare_dataset(train_file, val_file)
    model = load_model()
    trainer = train_model(model, train_dataset, val_dataset)

    # --- Log Evaluasi setelah Pelatihan --- #
    logger.info("Evaluasi model pada validation set:")
    eval_result = trainer.evaluate()
    logger.info(f"Akurasi setelah pelatihan: {eval_result['eval_accuracy']:.4f}")
    logger.info(f"F1 Score setelah pelatihan: {eval_result['eval_f1']:.4f}")
    logger.info(f"Precision setelah pelatihan: {eval_result['eval_precision']:.4f}")
    logger.info(f"Recall setelah pelatihan: {eval_result['eval_recall']:.4f}")
    logger.info(f"Eval Loss setelah pelatihan: {eval_result['eval_loss']:.4f}")

    save_model(trainer.model, tokenizer)