from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.jsonl"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts_v2"
MODEL_DIR = ARTIFACT_DIR / "models"
RUNTIME_DIR = ARTIFACT_DIR / "runtime"

TEMPLATE_BANK_PATH = ARTIFACT_DIR / "template_bank.json"

FIELDS = [
    "reference_no",
    "transaction_date",
    "account_no",
    "recipient_name",
    "total_amount",
]

# Target operasional
TARGET_FIELD_ACCURACY = 0.90
TARGET_LATENCY_SECONDS = 1.00

# Jika confidence di bawah threshold, field bisa ditandai needs_review
FIELD_CONFIDENCE_THRESHOLD = {
    "reference_no": 0.72,
    "transaction_date": 0.70,
    "account_no": 0.75,
    "recipient_name": 0.68,
    "total_amount": 0.75,
}

# Range nominal realistis untuk receipt transfer
MIN_AMOUNT = 1_000
MAX_AMOUNT = 100_000_000

# Resize batas atas untuk menyeimbangkan akurasi dan latency OCR.
OCR_MAX_WIDTH = 1100
