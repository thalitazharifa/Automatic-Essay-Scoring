import torch
from transformers import BertTokenizer, BertForSequenceClassification
# import os

# === Load Model dan Tokenizer ===
def load_grammar_checker_model(model_path="models/bert_cola_model"):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# === Fungsi Prediksi Grammar ===
def check_grammar(sentences, tokenizer, model, device):
    if isinstance(sentences, str):
        sentences = [sentences]

    inputs = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

    return predictions.cpu().tolist()