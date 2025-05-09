from modul_pengecekan_tatabahasa.main import load_grammar_checker_model, check_grammar
import spacy
from textblob import TextBlob
import jamspell

# === Load grammar checker model ===
tokenizer, model, device = load_grammar_checker_model()

# Load SpaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Load Jamspell corrector
jamspell_corrector = jamspell.TSpellCorrector()
jamspell_corrector.LoadLangModel('en.bin')  # Pastikan path benar

def fix_spelling_with_textblob(text):
    """Perbaikan ejaan menggunakan TextBlob"""
    return str(TextBlob(text).correct())

def fix_spelling_with_jamspell(text):
    """Perbaikan ejaan menggunakan Jamspell"""
    return jamspell_corrector.FixFragment(text)

def find_wrong_spelling_combined(text):
    """Gabungkan deteksi salah eja dari TextBlob dan Jamspell"""
    words = text.split()
    wrong_spelling = []

    for word in words:
        tb_corrected = str(TextBlob(word).correct())
        js_corrected = jamspell_corrector.FixFragment(word)
        
        best_correction = tb_corrected if tb_corrected != word else js_corrected
        if word != best_correction:
            wrong_spelling.append((word, best_correction))
    
    return wrong_spelling

def count_clauses(doc):
    """Menghitung jumlah klausa dari kalimat + klausa subordinat"""
    return sum(1 for sent in doc.sents for token in sent if token.dep_ in ["ccomp", "advcl", "relcl", "acl"]) + len(list(doc.sents))

def extract_grammar_features(essay: str):
    """Ekstraksi fitur tata bahasa dan ejaan dari teks esai"""
    doc = nlp(essay)
    sentences = list(doc.sents)

    # === Fitur Kalimat ===
    STg = len(sentences)  # Total kalimat
    sent_texts = [s.text.strip() for s in sentences]

    SWg_preds = check_grammar(sent_texts, tokenizer, model, device)
    SWg = sum(1 for pred in SWg_preds if pred == 1)  # Kalimat salah grammar

    # === Perbaikan ejaan (TextBlob + Jamspell) dan evaluasi ulang grammar ===
    fixed_sentences = [fix_spelling_with_jamspell(fix_spelling_with_textblob(s)) for s in sent_texts]
    SWg_new_preds = check_grammar(fixed_sentences, tokenizer, model, device)
    SWg_new = sum(1 for pred in SWg_new_preds if pred == 1)

    # === Fitur Kata ===
    tokens = [token.text for token in doc if token.is_alpha]
    WTs = len(tokens)  # Total kata

    wrong_spelling_details = []
    for sentence in sent_texts:
        wrong_spelling_details.extend(find_wrong_spelling_combined(sentence))
    Ws = len(wrong_spelling_details)  # Kata salah eja

    # === Fitur Klausa ===
    CTg = count_clauses(doc)  # Total klausa
    CWg = SWg  # Klausa salah grammar = jumlah kalimat salah
    CWg_new = SWg_new  # Setelah perbaikan ejaan

    # === Output fitur tata bahasa & ejaan ===
    grammar_features = {
        # Kalimat
        "STg": STg,
        "SWg": SWg,
        "SWg_new": SWg_new,
        "Grammar_Ratio_Sentence": SWg / STg if STg else 0,

        # Kata
        "WTs": WTs,
        "Ws": Ws,
        "Spelling_Ratio": Ws / WTs if WTs else 0,
        "Words_per_Sentence": WTs / STg if STg else 0,
        "Words_per_Clause": WTs / CTg if CTg else 0,

        # Klausa
        "CTg": CTg,
        "CWg": CWg,
        "CWg_new": CWg_new,
        "Grammar_Ratio_Clause": CWg / CTg if CTg else 0,
    }

    spelling_corrections = {
        "Wrong_Spelling_Details": wrong_spelling_details
    }

    return grammar_features, spelling_corrections