# src_v2/train_v2.py

import json
from pathlib import Path
import cv2
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_DIR,
    GROUND_TRUTH_PATH,
    MODEL_DIR,
    ARTIFACT_DIR,
    FIELDS,
)
from ocr_engine import ReceiptOCREngine
from image_preprocess import resize_for_speed, light_preprocess
from layout_parser import group_tokens_into_lines, normalize_text, normalize_number, build_page_text
from template_router import TemplateRouter
from candidate_generator import CandidateGenerator
from feature_builder import build_candidate_matrix


def load_ground_truth(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize_gt_value(field, value):
    """
    Normalisasi gold value agar comparable dengan kandidat.
    """
    if value is None:
        return None

    if field in ["account_no", "reference_no"]:
        return normalize_number(str(value)) if field == "account_no" else recompact(str(value))

    if field == "total_amount":
        return int(normalize_number(str(value))) if normalize_number(str(value)) else None

    if field == "recipient_name":
        return normalize_text(str(value))

    if field == "transaction_date":
        # Untuk training awal, cukup text contains.
        # Evaluasi final tetap pakai parser tanggal.
        return normalize_text(str(value))

    return normalize_text(str(value))


def recompact(text):
    import re
    return re.sub(r"\s+", "", str(text)).lower()


def is_candidate_correct(field, candidate_value, gt_value):
    """
    Matching kandidat terhadap ground truth.
    """
    if gt_value is None:
        return 0

    if candidate_value is None:
        return 0

    if field == "total_amount":
        try:
            return int(candidate_value) == int(gt_value)
        except Exception:
            return 0

    if field == "account_no":
        return int(normalize_number(str(candidate_value)) == normalize_number(str(gt_value)))

    if field == "reference_no":
        cand = recompact(candidate_value)
        gt = recompact(gt_value)

        return int(cand == gt or cand in gt or gt in cand)

    if field == "recipient_name":
        cand = normalize_text(candidate_value)
        gt = normalize_text(gt_value)

        if cand == gt:
            return 1

        cand_words = set(cand.split())
        gt_words = set(gt.split())

        return int(len(cand_words.intersection(gt_words)) >= max(1, min(2, len(gt_words))))

    if field == "transaction_date":
        cand = normalize_text(candidate_value)
        gt = normalize_text(gt_value)

        # Cukup longgar untuk training candidate.
        # Parser final akan distandarkan saat inference.
        return int(cand in gt or gt in cand or any(part in cand for part in gt.split()))

    return 0


def train_v2():
    """
    Training field-specific binary reranker.

    Output:
    - satu model per field
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_ground_truth(GROUND_TRUTH_PATH)

    ocr = ReceiptOCREngine()
    router = TemplateRouter()
    generator = CandidateGenerator()

    field_X = {field: [] for field in FIELDS}
    field_y = {field: [] for field in FIELDS}

    for row in rows:
        image_name = row["image"]
        gt = row["ground_truth"]

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(f"Skip missing image: {image_path}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        image, _ = resize_for_speed(image, max_width=1200)
        processed = light_preprocess(image)

        h, w = processed.shape[:2]

        tokens = ocr.run_ocr(processed)
        lines = group_tokens_into_lines(tokens)
        page_text = build_page_text(lines)

        route = router.detect_template(page_text)
        template_name = route["template"]

        candidates_by_field = generator.generate(lines)

        for field in FIELDS:
            candidates = candidates_by_field.get(field, [])

            if not candidates:
                continue

            X = build_candidate_matrix(candidates, w, h, template_name)

            gt_value = gt.get(field)

            y = [
                is_candidate_correct(field, c["value"], gt_value)
                for c in candidates
            ]

            field_X[field].append(X)
            field_y[field].extend(y)

    for field in FIELDS:
        if not field_X[field]:
            print(f"No training data for field: {field}")
            continue

        X = np.vstack(field_X[field])
        y = np.array(field_y[field], dtype=np.int32)

        if len(set(y)) < 2:
            print(f"Skip {field}, hanya punya satu kelas label.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        print("\n" + "=" * 80)
        print(f"FIELD: {field}")
        print(classification_report(y_val, pred, zero_division=0))

        try:
            print("AUC:", roc_auc_score(y_val, proba))
        except Exception:
            pass

        joblib.dump(model, MODEL_DIR / f"{field}.joblib")

    print(f"\nModel V2 disimpan ke: {MODEL_DIR}")


if __name__ == "__main__":
    train_v2()