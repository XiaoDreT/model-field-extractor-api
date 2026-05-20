import re
from typing import List, Dict


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_number(text: str) -> str:
    if text is None:
        return ""

    return re.sub(r"\D", "", str(text))


def sort_tokens_reading_order(tokens: List[Dict]) -> List[Dict]:
    """
    Mengurutkan token dari atas ke bawah, kiri ke kanan.
    """
    return sorted(tokens, key=lambda t: (t["cy"], t["cx"]))


def group_tokens_into_lines(tokens: List[Dict], y_threshold_ratio: float = 0.015):
    """
    Mengelompokkan token menjadi baris berdasarkan kedekatan koordinat y.

    Args:
        tokens: list token OCR
        y_threshold_ratio: toleransi jarak y relatif terhadap tinggi halaman

    Returns:
        list line:
        {
            "text": str,
            "tokens": list,
            "bbox": [x1, y1, x2, y2],
            "cx": float,
            "cy": float
        }
    """
    if not tokens:
        return []

    max_y = max(t["bbox"][3] for t in tokens)
    threshold = max_y * y_threshold_ratio

    sorted_tokens = sort_tokens_reading_order(tokens)

    lines = []

    for token in sorted_tokens:
        placed = False

        for line in lines:
            if abs(token["cy"] - line["cy"]) <= threshold:
                line["tokens"].append(token)
                line["cy"] = sum(t["cy"] for t in line["tokens"]) / len(line["tokens"])
                placed = True
                break

        if not placed:
            lines.append({
                "tokens": [token],
                "cy": token["cy"],
            })

    parsed_lines = []

    for line in lines:
        line_tokens = sorted(line["tokens"], key=lambda t: t["cx"])

        x1 = min(t["bbox"][0] for t in line_tokens)
        y1 = min(t["bbox"][1] for t in line_tokens)
        x2 = max(t["bbox"][2] for t in line_tokens)
        y2 = max(t["bbox"][3] for t in line_tokens)

        text = " ".join(t["text"] for t in line_tokens)
        text = re.sub(r"\s+", " ", text).strip()

        parsed_lines.append({
            "text": text,
            "norm_text": normalize_text(text),
            "tokens": line_tokens,
            "bbox": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })

    return sorted(parsed_lines, key=lambda x: x["cy"])


def build_page_text(lines: List[Dict]) -> str:
    """
    Menggabungkan semua line menjadi text halaman.
    """
    return "\n".join(line["text"] for line in lines)