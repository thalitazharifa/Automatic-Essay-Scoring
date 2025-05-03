import torch
from transformers import BertTokenizer, BertForSequenceClassification
from textblob import TextBlob
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Load model grammar checker
model_path = "./models/bert_cola_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_grammar(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
    return preds.cpu().numpy()

def extract_grammar_features(essay):
    sentences = sent_tokenize(essay)
    STg = len(sentences)
    sentence_preds = predict_grammar(sentences)
    
    sentence_analysis = []
    spelling_mistakes = []
    all_words = []

    for i, sentence in enumerate(sentences):
        corrected = str(TextBlob(sentence).correct())
        status = "Benar" if sentence_preds[i] == 0 else "Salah"
        sentence_analysis.append({
            "kalimat": sentence,
            "status": status,
            "koreksi": corrected
        })

        words = word_tokenize(sentence)
        all_words.extend(words)
        for w in words:
            corrected_word = str(TextBlob(w).correct())
            if corrected_word != w:
                spelling_mistakes.append({
                    "kata": w,
                    "koreksi": corrected_word,
                    "kalimat_ke": i + 1
                })

    corrected_sentences = [s["koreksi"] for s in sentence_analysis]
    SWg = sum([1 for s in sentence_analysis if s["status"] == "Salah"])
    SWg_new = sum(predict_grammar(corrected_sentences))

    WTg = len(all_words)
    Ws = len(spelling_mistakes)
    words_per_sent = WTg / STg if STg > 0 else 0

    clauses = re.split(r'[;,]', essay)
    CTg = len(clauses)
    clause_preds = predict_grammar(clauses)
    
    clause_analysis = []
    for i, clause in enumerate(clauses):
        corrected = str(TextBlob(clause).correct())
        status = "Benar" if clause_preds[i] == 0 else "Salah"
        clause_analysis.append({
            "klausa": clause.strip(),
            "status": status,
            "koreksi": corrected
        })

    corrected_clauses = [c["koreksi"] for c in clause_analysis]
    CWg = sum([1 for c in clause_analysis if c["status"] == "Salah"])
    CWg_new = sum(predict_grammar(corrected_clauses))
    words_per_clause = WTg / CTg if CTg > 0 else 0

    return {
        "summary": {
            "STg": STg,
            "SWg": SWg,
            "SWg_new": SWg_new,
            "WTg": WTg,
            "Ws": Ws,
            "CTg": CTg,
            "CWg": CWg,
            "CWg_new": CWg_new,
            "WTg/STg": round(words_per_sent, 2),
            "WTg/CTg": round(words_per_clause, 2),
            "SWg/STg": round(SWg / STg, 2) if STg else 0,
            "CWg/CTg": round(CWg / CTg, 2) if CTg else 0,
            "Ws/WTg": round(Ws / WTg, 2) if WTg else 0
        },
        "per_kalimat": sentence_analysis,
        "per_klausa": clause_analysis,
        "kesalahan_ejaan": spelling_mistakes
    }