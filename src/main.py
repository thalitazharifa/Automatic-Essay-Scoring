import pandas as pd
import logging
from sklearn.metrics import cohen_kappa_score
from modul_penilaian_struktur.main import load_structure_model, predict_structure_score, evaluate_qwk
from modul_pengecekan_tatabahasa.main import load_grammar_checker_model
from modul_pengecekan_tatabahasa.fitur_tata_bahasa import extract_grammar_features
from modul_penilaian_tatabahasa.main import predict_grammar_score, load_regression_model
import spacy

# === Logging Setup ===
log_file_path = 'src/data/log_validation.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

# === Load Models ===
logger.info("Loading structure model...")
structure_tokenizer, structure_model, structure_device = load_structure_model()
logger.info("Structure model loaded successfully.")

logger.info("Loading grammar model...")
grammar_tokenizer, grammar_model, grammar_device = load_grammar_checker_model()
logger.info("Grammar model loaded successfully.")

logger.info("Loading regression model...")
regression_model = load_regression_model()
logger.info("Regression model loaded successfully.")

logger.info("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")
logger.info("SpaCy model loaded successfully.")

# === Evaluation Function ===
def evaluate_essay(essay):
    logger.info("Evaluating essay...")
    structure_score = predict_structure_score([essay], structure_model, structure_tokenizer, structure_device)
    grammar_features, _ = extract_grammar_features(essay)
    grammar_score = predict_grammar_score(grammar_features, regression_model)
    logger.info("Essay evaluated successfully.")
    return structure_score, grammar_score, grammar_features

# === Process Essays from CSV ===
def process_essays(input_csv, output_csv):
    logger.info(f"Processing essays from {input_csv}...")
    df = pd.read_csv(input_csv)
    results = []

    structure_true, structure_pred = [], []
    grammar_true, grammar_pred = [], []

    if 'essay' not in df.columns:
        raise ValueError("Kolom 'essay' tidak ditemukan dalam file input.")

    for i, essay in enumerate(df['essay']):
        logger.info(f"Processing essay {i+1}/{len(df)}...")

        try:
            structure_score, grammar_score, grammar_features = evaluate_essay(essay)
        except Exception as e:
            logger.error(f"Gagal memproses essay ke-{i+1}: {e}")
            structure_score, grammar_score, grammar_features = None, None, {}

        nilai_struktur_rater = df.loc[i, 'skor_struktur_normalized'] if 'skor_struktur_normalized' in df.columns and not pd.isna(df.loc[i, 'skor_struktur_normalized']) else None
        nilai_grammar_rater = df.loc[i, 'skor_tata_bahasa_normalized'] if 'skor_tata_bahasa_normalized' in df.columns and not pd.isna(df.loc[i, 'skor_tata_bahasa_normalized']) else None

        results.append({
            'essay': essay,
            'nilai_struktur': structure_score,
            'nilai_struktur_rater': nilai_struktur_rater,
            'nilai_tata_bahasa': grammar_score,
            'nilai_tata_bahasa_rater': nilai_grammar_rater,
            'fitur_tata_bahasa': str(grammar_features),
        })

        if nilai_struktur_rater is not None:
            structure_true.append(nilai_struktur_rater)
            structure_pred.append(structure_score)
        if nilai_grammar_rater is not None:
            grammar_true.append(nilai_grammar_rater)
            grammar_pred.append(grammar_score)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Hasil disimpan ke {output_csv}")
    print(f"Hasil disimpan ke {output_csv}")

    if structure_true and structure_pred:
        structure_qwk = evaluate_qwk(structure_true, structure_pred)
        logger.info(f"Structure QWK: {structure_qwk:.4f}")
        print(f"Structure QWK: {structure_qwk:.4f}")

    if grammar_true and grammar_pred:
        grammar_qwk = cohen_kappa_score(grammar_true, grammar_pred, weights='quadratic')
        logger.info(f"Grammar QWK: {grammar_qwk:.4f}")
        print(f"Grammar QWK: {grammar_qwk:.4f}")

# === Main Execution ===
if __name__ == "__main__":
    input_file = 'data/pre-processing-data/validation/validation_data.csv'
    output_file = '/data/output_results_log.csv'
    logger.info("Starting the essay evaluation process...")
    process_essays(input_file, output_file)
    logger.info("Essay evaluation process completed.")