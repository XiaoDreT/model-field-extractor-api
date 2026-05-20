import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

try:
    from .config import GROUND_TRUTH_PATH, IMAGE_DIR, FIELDS
    from .inference_v2 import ReceiptFieldExtractorV2
    from .layout_parser import normalize_number, normalize_text
except ImportError:
    from config import GROUND_TRUTH_PATH, IMAGE_DIR, FIELDS
    from inference_v2 import ReceiptFieldExtractorV2
    from layout_parser import normalize_number, normalize_text


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_field_value(field: str, value):
    if value is None:
        return None

    if field == "account_no":
        digits = normalize_number(str(value))
        return digits or None

    if field == "reference_no":
        return normalize_text(str(value)).replace(" ", "") or None

    if field == "recipient_name":
        return normalize_text(str(value)) or None

    if field == "total_amount":
        digits = normalize_number(str(value))
        return int(digits) if digits else None

    if field == "transaction_date":
        text = normalize_text(str(value))
        if not text:
            return None

        # Standarkan agar toleran format input berbeda.
        try:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M")
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return text

    return normalize_text(str(value))


def is_match(field: str, pred, gt) -> bool:
    if gt is None:
        return pred is None

    if pred is None:
        return False

    if field == "reference_no":
        pred_c = normalize_text(str(pred)).replace(" ", "")
        gt_c = normalize_text(str(gt)).replace(" ", "")
        return pred_c == gt_c or pred_c in gt_c or gt_c in pred_c

    if field == "recipient_name":
        pred_n = normalize_text(str(pred))
        gt_n = normalize_text(str(gt))
        if pred_n == gt_n:
            return True

        pred_words = set(pred_n.split())
        gt_words = set(gt_n.split())
        if not pred_words or not gt_words:
            return False

        overlap = len(pred_words.intersection(gt_words))
        return overlap >= min(2, len(gt_words))

    return normalize_field_value(field, pred) == normalize_field_value(field, gt)


def evaluate(limit=None):
    rows = load_rows(Path(GROUND_TRUTH_PATH))
    if limit is not None:
        rows = rows[:limit]

    extractor = ReceiptFieldExtractorV2()

    correct = {field: 0 for field in FIELDS}
    total = {field: 0 for field in FIELDS}
    latencies = []
    failures = []

    for row in rows:
        image_name = row["image"]
        gt = row["ground_truth"]
        image_path = Path(IMAGE_DIR) / image_name

        output = extractor.predict(str(image_path), return_meta=True)
        pred = output["data"]
        latencies.append(float(output.get("latency_seconds", 0.0)))

        row_failed_fields = []

        for field in FIELDS:
            gt_value = gt.get(field)
            if gt_value is None:
                continue

            total[field] += 1

            if is_match(field, pred.get(field), gt_value):
                correct[field] += 1
            else:
                row_failed_fields.append(field)

        if row_failed_fields:
            failures.append({
                "image": image_name,
                "failed_fields": row_failed_fields,
                "prediction": pred,
                "ground_truth": gt,
                "latency_seconds": output.get("latency_seconds", 0.0),
            })

    per_field_acc = {}
    for field in FIELDS:
        per_field_acc[field] = (correct[field] / total[field]) if total[field] else 0.0

    total_correct = sum(correct.values())
    total_count = sum(total.values())
    overall_acc = (total_correct / total_count) if total_count else 0.0

    lat_sorted = sorted(latencies)
    p95_idx = max(0, int(0.95 * len(lat_sorted)) - 1)

    metrics = {
        "samples": len(rows),
        "per_field": {
            field: {
                "correct": correct[field],
                "total": total[field],
                "accuracy": round(per_field_acc[field], 4),
            }
            for field in FIELDS
        },
        "overall_accuracy": round(overall_acc, 4),
        "latency": {
            "mean": round(statistics.mean(latencies), 4) if latencies else 0.0,
            "p95": round(lat_sorted[p95_idx], 4) if latencies else 0.0,
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "failure_count": len(failures),
        "top_failures": sorted(failures, key=lambda x: len(x["failed_fields"]), reverse=True)[:10],
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cetakia V2 extractor")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N samples")
    parser.add_argument("--json", action="store_true", help="Print raw JSON metrics")
    args = parser.parse_args()

    metrics = evaluate(limit=args.limit)

    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    print("=" * 84)
    print(f"Samples: {metrics['samples']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(
        "Latency mean/p95/max: "
        f"{metrics['latency']['mean']:.4f}s / "
        f"{metrics['latency']['p95']:.4f}s / "
        f"{metrics['latency']['max']:.4f}s"
    )
    print("-" * 84)

    for field, stat in metrics["per_field"].items():
        print(
            f"{field:16s} "
            f"acc={stat['accuracy']:.4f} "
            f"({stat['correct']}/{stat['total']})"
        )

    print("-" * 84)
    print(f"Failure rows: {metrics['failure_count']}")


if __name__ == "__main__":
    main()
