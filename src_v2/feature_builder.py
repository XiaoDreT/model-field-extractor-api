import re
import numpy as np

try:
    from .layout_parser import normalize_text, normalize_number
    from .candidate_generator import (
        REFERENCE_ANCHORS,
        DATE_ANCHORS,
        ACCOUNT_ANCHORS,
        NAME_ANCHORS,
        AMOUNT_ANCHORS,
    )
except ImportError:
    from layout_parser import normalize_text, normalize_number
    from candidate_generator import (
        REFERENCE_ANCHORS,
        DATE_ANCHORS,
        ACCOUNT_ANCHORS,
        NAME_ANCHORS,
        AMOUNT_ANCHORS,
    )


FIELD_TO_ID = {
    "reference_no": 0,
    "transaction_date": 1,
    "account_no": 2,
    "recipient_name": 3,
    "total_amount": 4,
}

SOURCE_TO_ID = {
    "direct_reference_pattern": 0,
    "anchor_reference": 1,
    "direct_date_pattern": 2,
    "anchor_date": 3,
    "direct_account_pattern": 4,
    "anchor_account": 5,
    "direct_name_pattern": 6,
    "anchor_name": 7,
    "direct_amount_pattern": 8,
    "anchor_amount": 9,
}

TEMPLATE_TO_ID = {
    None: 0,
    "bca": 1,
    "bni": 2,
    "livin_mandiri": 3,
    "byond_bsi": 4,
    "seabank": 5,
    "blu_bca": 6,
}


def count_anchor_hits(text, anchors):
    norm = normalize_text(text)
    return sum(1 for anchor in anchors if anchor in norm)


def candidate_to_features(candidate, image_width, image_height, template_name=None):
    """
    Mengubah satu kandidat field menjadi feature vector numeric.

    Args:
        candidate: dict kandidat
        image_width: lebar image
        image_height: tinggi image
        template_name: hasil template router

    Returns:
        np.array shape (n_features,)
    """
    value = str(candidate["value"])
    line_text = candidate.get("line_text", "")

    x1, y1, x2, y2 = candidate["bbox"]

    digits = normalize_number(value)

    digit_count = sum(c.isdigit() for c in value)
    alpha_count = sum(c.isalpha() for c in value)
    symbol_count = sum(not c.isalnum() and not c.isspace() for c in value)

    field = candidate["field"]
    source = candidate.get("source")

    features = [
        FIELD_TO_ID.get(field, -1),
        SOURCE_TO_ID.get(source, -1),
        TEMPLATE_TO_ID.get(template_name, 0),

        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
        ((x1 + x2) / 2) / image_width,
        ((y1 + y2) / 2) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,

        len(value),
        len(normalize_text(value)),
        len(digits),
        digit_count,
        alpha_count,
        symbol_count,

        int("rp" in normalize_text(line_text)),
        int("idr" in normalize_text(line_text)),
        int("*" in value),

        count_anchor_hits(line_text, REFERENCE_ANCHORS),
        count_anchor_hits(line_text, DATE_ANCHORS),
        count_anchor_hits(line_text, ACCOUNT_ANCHORS),
        count_anchor_hits(line_text, NAME_ANCHORS),
        count_anchor_hits(line_text, AMOUNT_ANCHORS),
    ]

    return np.array(features, dtype=np.float32)


def build_candidate_matrix(candidates, image_width, image_height, template_name=None):
    if not candidates:
        return np.empty((0, 24), dtype=np.float32)

    return np.vstack([
        candidate_to_features(c, image_width, image_height, template_name)
        for c in candidates
    ])
