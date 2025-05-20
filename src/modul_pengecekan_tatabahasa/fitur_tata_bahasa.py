from modul_pengecekan_tatabahasa.main import load_grammar_checker_model, check_grammar
from textblob import TextBlob
import spacy
import jamspell

# === Load grammar checker model ===
tokenizer, model, device = load_grammar_checker_model()

# Load SpaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Load Jamspell corrector
jamspell_corrector = jamspell.TSpellCorrector()
jamspell_corrector.LoadLangModel('en.bin') 

def fix_spelling_with_textblob(text):
    """Perbaikan ejaan menggunakan TextBlob"""
    return str(TextBlob(text).correct())

def fix_spelling_with_jamspell(text):
    """Perbaikan ejaan menggunakan Jamspell"""
    return jamspell_corrector.FixFragment(text)

def find_wrong_spelling_combined_context(text):
    """Deteksi salah eja berbasis konteks kalimat dan cek grammar hasil koreksi"""
    wrong_spelling = []

    original_score = check_grammar([text], tokenizer, model, device)[0]

    # Koreksi pakai TextBlob
    tb_fixed = fix_spelling_with_textblob(text)
    tb_score = check_grammar([tb_fixed], tokenizer, model, device)[0]

    # Koreksi pakai Jamspell
    js_fixed = fix_spelling_with_jamspell(text)
    js_score = check_grammar([js_fixed], tokenizer, model, device)[0]

    # Ambil versi terbaik berdasarkan skor grammar
    if tb_score > original_score or js_score > original_score:
        if tb_score >= js_score and tb_fixed != text:
            wrong_spelling = [(w, str(TextBlob(w).correct())) for w in text.split() if w != str(TextBlob(w).correct())]
        elif js_fixed != text:
            wrong_spelling = [(w, jamspell_corrector.FixFragment(w)) for w in text.split() if w != jamspell_corrector.FixFragment(w)]

    return wrong_spelling

def count_clauses(doc):
    """Menghitung jumlah klausa dari kalimat + klausa subordinat"""
    return sum(1 for sent in doc.sents for token in sent if token.dep_ in ["ccomp", "advcl", "relcl", "acl"]) + len(list(doc.sents))

def save_grammar_log_to_txt(essay_id, essay_text, grammar_features, sent_texts, clause_texts, clause_owner, wrong_spelling_details, grammar_preds, filepath="log_grammar.txt"):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"=== Essay {essay_id} ===\n")
        f.write(f"Jumlah Kalimat: {grammar_features['STg']}, Jumlah Klausa: {grammar_features['CTg']}\n\n")

        for i, kalimat in enumerate(sent_texts):
            status = "Benar Grammar" if grammar_preds[i] == 1 else "Salah Grammar"
            f.write(f"[Kalimat {i+1}]: {kalimat} [{status}]\n")
            klausa_terkait = [clause_texts[j] for j, owner in enumerate(clause_owner) if owner == i]
            for k, klausa in enumerate(klausa_terkait):
                clause_status = "Benar Grammar" if grammar_preds[len(sent_texts) + k] == 1 else "Salah Grammar"
                f.write(f"  - Klausa {k+1}: {klausa} [{clause_status}]\n")
            f.write("\n")

        f.write("--- Salah Eja & Koreksi ---\n")
        for asli, koreksi in wrong_spelling_details:
            f.write(f"{asli} → {koreksi}\n")

        f.write("\n\n")

def extract_grammar_features(essay: str, essay_id="1", log_path="logs/training/modul_penilaian_tata_bahasa/log_grammar.txt"):
    """Ekstraksi fitur tata bahasa dan ejaan dari teks esai + logging hasil"""
    doc = nlp(essay)
    sentences = list(doc.sents)
    sent_texts = [s.text.strip() for s in sentences]
    STg = len(sentences)  # Total kalimat

    # === Ekstrak klausa dari setiap kalimat dan simpan info posisi klausa ===
    clause_texts = []
    clause_owner = []  # untuk melacak klausa berasal dari kalimat ke-i
    for i, sent in enumerate(sentences):
        for token in sent:
            if token.dep_ in ["ccomp", "advcl", "relcl", "acl"]:
                clause = token.subtree
                clause_text = " ".join([t.text for t in clause])
                clause_texts.append(clause_text.strip())
                clause_owner.append(i)

    CTg = STg + len(clause_texts)  # Total klausa = kalimat + klausa subordinat

    # === Gabungkan kalimat + klausa untuk grammar checking ===
    all_clauses = sent_texts + clause_texts
    grammar_preds = check_grammar(all_clauses, tokenizer, model, device)

    # Kalimat salah grammar (0 = salah grammar, 1 = grammar benar)
    SWg = sum(1 for pred in grammar_preds[:STg] if pred == 0)

    # Cek apakah klausa yang salah menjadikan kalimat induknya salah juga
    wrong_clause_indices = [idx for idx, pred in enumerate(grammar_preds[STg:]) if pred == 0]
    wrong_clause_owners = set(clause_owner[i] for i in wrong_clause_indices)

    SWg_with_clause_check = 0
    for i in range(STg):
        if grammar_preds[i] == 0 or i in wrong_clause_owners:
            SWg_with_clause_check += 1

    # Total salah grammar seluruh kalimat dan klausa
    CWg = sum(1 for pred in grammar_preds if pred == 0)

    # === Perbaikan ejaan (TextBlob + Jamspell) ===
    fixed_sentences = [fix_spelling_with_jamspell(fix_spelling_with_textblob(s)) for s in sent_texts]
    fixed_clauses = [fix_spelling_with_jamspell(fix_spelling_with_textblob(c)) for c in clause_texts]
    all_fixed_clauses = fixed_sentences + fixed_clauses

    grammar_preds_fixed = check_grammar(all_fixed_clauses, tokenizer, model, device)

    SWg_new = sum(1 for pred in grammar_preds_fixed[:STg] if pred == 0)
    CWg_new = sum(1 for pred in grammar_preds_fixed if pred == 0)

    # === Fitur Kata dan Ejaan ===
    tokens = [token.text for token in doc if token.is_alpha]
    WTs = len(tokens)

    wrong_spelling_details = []
    for sentence in sent_texts:
        wrong_spelling_details.extend(find_wrong_spelling_combined_context(sentence))

    Ws = len(wrong_spelling_details)

    grammar_features = {
        "STg": STg,
        "SWg": SWg_with_clause_check,
        "SWg_new": SWg_new,
        "Grammar_Ratio_Sentence": SWg_with_clause_check / STg if STg else 0,

        "WTs": WTs,
        "Ws": Ws,
        "Spelling_Ratio": Ws / WTs if WTs else 0,
        "Words_per_Sentence": WTs / STg if STg else 0,
        "Words_per_Clause": WTs / CTg if CTg else 0,

        "CTg": CTg,
        "CWg": CWg,
        "CWg_new": CWg_new,
        "Grammar_Ratio_Clause": CWg / CTg if CTg else 0,
    }

    spelling_corrections = {
        "Wrong_Spelling_Details": wrong_spelling_details
    }

    # === Logging ===
    # Save grammar log after extracting features
    save_grammar_log_to_txt(
        essay_id=essay_id,
        essay_text=essay,
        grammar_features=grammar_features,
        sent_texts=sent_texts,
        clause_texts=clause_texts,
        clause_owner=clause_owner,
        wrong_spelling_details=wrong_spelling_details,
        grammar_preds=grammar_preds,
        filepath=log_path
    )

    return grammar_features, spelling_corrections