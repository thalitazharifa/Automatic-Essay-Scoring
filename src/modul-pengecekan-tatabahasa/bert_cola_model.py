import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# === 1. Gunakan GPU jika tersedia ===
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Menggunakan perangkat: {device}")

# === 2. Load CoLA Dataset ===
def load_data():
    return load_dataset("glue", "cola")

# === 3. Tokenisasi Dataset ===
def preprocess_data(example, tokenizer):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

def prepare_dataset():
    dataset = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset, tokenizer

# === 4. Load Pre-trained BERT Model ===
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    return model

# === 5. Fungsi Evaluasi (Metrik) ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# === 6. Fine-tuning Model ===
def train_model(model, dataset):
    training_args = TrainingArguments(
        output_dir="./models/bert_cola_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        report_to=[],  # Disable reporting (WandB, TensorBoard, dll)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    print("Melatih model... (Silakan tunggu)")
    trainer.train()
    print("Model telah dilatih!")

    # Evaluasi model terbaik setelah training
    metrics = trainer.evaluate()
    print(f"Model terbaik: Accuracy = {metrics['eval_accuracy']:.4f}, F1 = {metrics['eval_f1']:.4f}")
    
    return trainer

# === 7. Simpan Model ke HuggingFace + ONNX ===
def save_model(model, tokenizer, save_path="./models/bert_cola_model"):
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # ONNX Export
    dummy_input = tokenizer("This is a test sentence.", return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        f"{save_path}/model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=16
    )

    print(f"Model disimpan ke: {save_path}")
    print(f"Model universal tersedia di: {save_path}/model.onnx")

# === Main Execution ===
if __name__ == "__main__":
    dataset, tokenizer = prepare_dataset()
    model = load_model()
    trainer = train_model(model, dataset)
    save_model(trainer.model, tokenizer)