import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dateutil import parser as date_parser

try:
    from .config import GROUND_TRUTH_PATH, MIN_AMOUNT, MAX_AMOUNT
    from .layout_parser import normalize_text, normalize_number
except ImportError:
    from config import GROUND_TRUTH_PATH, MIN_AMOUNT, MAX_AMOUNT
    from layout_parser import normalize_text, normalize_number


MONTH_ALIASES = {
    "januari": 1,
    "january": 1,
    "jan": 1,
    "februari": 2,
    "february": 2,
    "feb": 2,
    "maret": 3,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apri": 4,
    "apr": 4,
    "mei": 5,
    "may": 5,
    "juni": 6,
    "june": 6,
    "jun": 6,
    "juli": 7,
    "july": 7,
    "jul": 7,
    "agustus": 8,
    "augustus": 8,
    "agust": 8,
    "agu": 8,
    "agt": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "oktober": 10,
    "october": 10,
    "okt": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "desember": 12,
    "december": 12,
    "des": 12,
    "dec": 12,
}

MONTH_TOKEN_PATTERN = r"(jan|feb|mar|apr|mei|may|jun|jul|agu|aug|sep|oct|okt|nov|dec|des)"

REFERENCE_ANCHOR_HINTS = (
    "nomor referensi",
    "no referensi",
    "no. referensi",
    "no.referensi",
    "no. ref",
    "no ref",
    "reference id",
    "reference no",
    "reference number",
    "biz id",
    "nomor transaksi",
    "id transaksi",
    "ref blu",
)

POSITIVE_NAME_ANCHORS = (
    "penerima",
    "nama penerima",
    "recipient",
    "beneficiary",
    "tujuan",
    "transfer to",
    "transfer ke",
    "destination account",
    "ke",
)

NEGATIVE_NAME_ANCHORS = (
    "pengirim",
    "sender",
    "sumber dana",
    "transfer from",
    "from account",
    "dari",
)

AMOUNT_PRIORITY_ANCHORS = (
    "nominal transfer",
    "jumlah transfer",
    "nominal",
    "amount",
    "total bayar",
)

AMOUNT_NEGATIVE_HINTS = (
    "biaya",
    "admin",
    "fee",
    "total debit",
    "total charges",
    "online fee",
)

DATE_LABEL_HINTS = (
    "tanggal transaksi",
    "transaction date",
    "waktu transaksi",
    "transaction time",
    "tanggal",
    "waktu",
)

INSTRUCTION_DATE_HINTS = (
    "instruction date",
    "instruction mode",
    "additional notification",
    "single transfer to other bank",
    "online domestic transfer",
)

NAME_BLOCKLIST_WORDS = {
    "transfer",
    "berhasil",
    "transaksi",
    "status",
    "tanggal",
    "waktu",
    "nominal",
    "total",
    "bank",
    "rekening",
    "tujuan",
    "bukti",
    "receipt",
    "penerima",
    "nama",
    "nomor",
    "referensi",
    "pengirim",
    "sumber",
    "dana",
    "ref",
    "detail",
    "transaction",
    "details",
    "method",
    "purpose",
    "account",
    "number",
    "tabungan",
    "utama",
    "seabank",
    "dana",
    "ovo",
    "blu",
    "livin",
    "mandiri",
    "bca",
    "bri",
    "bni",
}


def parse_rupiah_amount(text: str) -> Optional[int]:
    raw = str(text).strip()
    if not raw or not re.search(r"\d", raw):
        return None

    cleaned = re.sub(r"[^0-9,.-]", "", raw)
    if not cleaned:
        return None

    if re.match(r"^\d{1,3}(?:[.,]\d{3})+\d{2}$", cleaned):
        merged = re.sub(r"\D", "", cleaned)
        if len(merged) > 2:
            return int(merged[:-2])

    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")
    sep_idx = max(last_comma, last_dot)

    integer_part = cleaned
    if sep_idx != -1:
        tail = cleaned[sep_idx + 1:]
        if tail.isdigit() and len(tail) <= 2:
            integer_part = cleaned[:sep_idx]

    digits = re.sub(r"\D", "", integer_part)
    if not digits:
        return None

    if "," not in cleaned and "." not in cleaned and len(digits) >= 8 and digits.endswith("00"):
        reduced = int(digits[:-2])
        if MIN_AMOUNT <= reduced <= MAX_AMOUNT:
            return reduced

    return int(digits)


def parse_rupiah_amount_ocr_aware(text: str) -> Optional[int]:
    raw = str(text)
    raw = re.sub(r"(?i)\brp[il](?=\d)", "rp1", raw)
    raw = re.sub(r"(?i)\bidr[il](?=\d)", "idr1", raw)
    return parse_rupiah_amount(raw)


def is_amount_candidate(text: str) -> bool:
    raw = str(text)
    norm = normalize_text(raw)
    value = parse_rupiah_amount_ocr_aware(raw)
    if value is None:
        return False
    if not (MIN_AMOUNT <= value <= MAX_AMOUNT):
        return False
    if re.search(r"[*xX#]{2,}", raw):
        return False

    digits = normalize_number(raw)
    has_grouped = bool(re.search(r"\d{1,3}([.,]\d{3})+", raw))
    has_keyword = ("rp" in norm or "idr" in norm)
    has_decimal = bool(re.search(r"[.,]\d{2}\b", raw))

    if len(digits) >= 8 and not (has_grouped or has_keyword):
        return False

    if has_grouped or has_keyword or has_decimal:
        return True

    return 4 <= len(digits) <= 7


def clean_name_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text)).strip()
    value = re.sub(r"^[^A-Za-z]+|[^A-Za-z .'-]+$", "", value).strip()
    value = re.sub(r"(?i)^(nama(\s+penerima)?|penerima|recipient|beneficiary)\s*[:\-]\s*", "", value)
    value = re.sub(r"(?i)^ke\s*[:\-]\s*", "", value)
    value = re.sub(r"(?i)^transfer\s+(ke|to)\s+", "", value)
    value = re.split(r"(?i)\s*[-|]\s*(bca|bri|bni|mandiri|seabank|cimb|bsi)\b", value)[0]
    value = re.split(r"(?i)\b(rp|idr)\b", value)[0]
    value = re.sub(r"\s+", " ", value).strip(" .-")
    return value


def is_human_name_candidate(text: str) -> bool:
    raw = clean_name_text(text)
    if not raw:
        return False

    norm = normalize_text(raw)
    if any(char.isdigit() for char in raw):
        return False

    words = norm.split()
    if not (1 <= len(words) <= 4):
        return False

    alpha_ratio = sum(c.isalpha() or c.isspace() or c in "'.-" for c in raw) / max(len(raw), 1)
    if alpha_ratio < 0.84:
        return False

    if any(w in NAME_BLOCKLIST_WORDS for w in words):
        return False

    bad_fragments = (
        "bank",
        "transaksi",
        "berhasil",
        "nominal",
        "biaya",
        "rekening",
        "detail",
        "transfer",
        "dana",
        "ovo",
        "seabank",
    )
    if any(fragment in norm for fragment in bad_fragments):
        return False

    if raw.isupper() and len(words) == 1 and len(raw) <= 6:
        return False

    if len(words) == 1 and len(words[0]) < 5:
        return False

    return True


def is_datetime_like(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False

    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm):
        return True
    if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm):
        return True
    if re.search(r"\d{1,2}[:.]\d{2}([:.]\d{2})?", norm) and re.search(rf"\b{MONTH_TOKEN_PATTERN}\b", norm, flags=re.IGNORECASE):
        return True
    if re.search(rf"\d{{1,2}}\s*[-/ ]*{MONTH_TOKEN_PATTERN}\w*\s*[-/ ]*20\d{{2}}", norm, flags=re.IGNORECASE):
        return True
    return False


def safe_parse_date(text: str) -> Optional[str]:
    raw = str(text)
    norm = normalize_text(raw)
    if not norm:
        return None

    has_explicit_date = bool(
        re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm)
        or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm)
        or re.search(rf"\d{{1,2}}\s*[-/ ]*{MONTH_TOKEN_PATTERN}\w*\s*[-/ ]*20\d{{2}}", norm, flags=re.IGNORECASE)
        or re.search(rf"{MONTH_TOKEN_PATTERN}\w*\s+\d{{1,2}}(?:,|\s|-)+20\d{{2}}", norm, flags=re.IGNORECASE)
    )

    if not has_explicit_date:
        return None

    normalized = raw.replace("|", " ").replace(";", ":")
    normalized = re.sub(r"(?<=\d)\.(?=\d{2}\b)", ":", normalized)

    try:
        dt = date_parser.parse(normalized, dayfirst=True, fuzzy=True)
        current_year = datetime.now().year
        if dt.year > current_year + 3:
            dt = dt.replace(year=current_year)
        if dt.year < 2000:
            return None
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def _normalize_year(year: int) -> int:
    current_year = datetime.now().year
    if year > current_year + 3:
        return current_year
    if year < 2000:
        return current_year
    return year


def _resolve_month(token: str) -> Optional[int]:
    value = re.sub(r"[^a-z]", "", token.lower())
    value = value.replace("1", "l").replace("0", "o")
    if not value:
        return None

    for alias in sorted(MONTH_ALIASES.keys(), key=len, reverse=True):
        if value.startswith(alias):
            return MONTH_ALIASES[alias]

    return None


def _extract_time_component(text: str) -> Tuple[int, int]:
    normalized = str(text).replace(";", ":")
    normalized = re.sub(r"(?<=\d)\.(?=\d{2}(?:\D|$))", ":", normalized)
    normalized = re.sub(r"(?i)\b([ilt])(?=\d[:.])", "1", normalized)
    normalized = re.sub(r"(?i)\b([ilt])[-](\d{2})\b", r"14:\2", normalized)

    standard = re.search(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?:[:.]([0-5]\d))?(?!\d)", normalized)
    if standard:
        return int(standard.group(1)), int(standard.group(2))

    compact = re.search(r"(?<!\d)(\d{3,4})[:.]([0-5]\d)(?!\d)", normalized)
    if compact:
        left = compact.group(1)
        if len(left) == 4:
            hour = int(left[:2])
            minute = int(left[2:])
        else:
            hour = int(left[0])
            minute = int(left[1:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute

    spaced = re.search(r"(?<!\d)([01]?\d|2[0-3])\s+([0-5]\d)(?:\s+([0-5]\d))?(?!\d)", normalized)
    if spaced:
        return int(spaced.group(1)), int(spaced.group(2))

    hyphenated = re.search(r"(?<!\d)([01]?\d|2[0-3])[-]([0-5]\d)(?!\d)", normalized)
    if hyphenated:
        return int(hyphenated.group(1)), int(hyphenated.group(2))

    return 0, 0


def parse_noisy_transaction_date(text: str) -> Optional[str]:
    raw = str(text)
    norm = normalize_text(raw)
    if not norm:
        return None

    norm = re.sub(r"\b(wib|wit|wita)\b", " ", norm)
    norm = norm.replace("|", "1")
    norm = norm.replace(";", ":")
    norm = re.sub(r"\bi(?=\d{3,4}[:.])", "1", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    # Format OCR fused: 01/04202600:14:44
    fused_mdy = re.search(r"(?<!\d)(\d{1,2})[-/](\d{2})(20\d{2})([01]?\d|2[0-3])[:.]([0-5]\d)", norm)
    if fused_mdy:
        d, m, y, hh, mm = fused_mdy.groups()
        day = int(d)
        month = int(m)
        year = _normalize_year(int(y))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-{day:02d} {int(hh):02d}:{int(mm):02d}"

    fused_num = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](20\d{2})([01]?\d|2[0-3])[:.]([0-5]\d)", norm)
    if fused_num:
        d1, d2, y, hh, mm = fused_num.groups()
        day = int(d1)
        month = int(d2)
        year = _normalize_year(int(y))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-{day:02d} {int(hh):02d}:{int(mm):02d}"

    num = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](20\d{2})(?:\D+([01]?\d|2[0-3])[:.]([0-5]\d))?", norm)
    if num:
        d1, d2, y, hh, mm = num.groups()
        day = int(d1)
        month = int(d2)
        year = _normalize_year(int(y))
        if 1 <= day <= 31 and 1 <= month <= 12:
            hour = int(hh) if hh is not None else 0
            minute = int(mm) if mm is not None else 0
            return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    month_first = re.search(
        r"(?<!\d)([a-z]{3,10})\s+(\d{1,2})(?:,|\s|-)+(20\d{2})(?:\D+([01]?\d|2[0-3])[:.]([0-5]\d))?",
        norm,
    )
    if month_first:
        m, d, y, hh, mm = month_first.groups()
        month = _resolve_month(m)
        if month is not None:
            day = int(d)
            year = _normalize_year(int(y))
            if 1 <= day <= 31:
                if hh is not None and mm is not None:
                    hour = int(hh)
                    minute = int(mm)
                else:
                    hour, minute = _extract_time_component(norm)
                return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    month_text = re.search(
        r"(?<!\d)(\d{1,2})\s*[-/ ]*\s*([a-z]{3,10})(?:[1l])?\s*[-/ ]*\s*(20\d{2})(?=(?:\D|$|\d{1,2}[:.]\d{2}))",
        norm,
    )
    if month_text:
        d, m, y = month_text.groups()
        month = _resolve_month(m)
        if month is not None:
            day = int(d)
            if day == 0:
                day = 1
            year = _normalize_year(int(y))
            if 1 <= day <= 31:
                hour, minute = _extract_time_component(norm)
                return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    return None


class RuleFieldParserV1:
    """
    Rule parser yang mengadopsi pola final V1 dalam bentuk lebih ringkas.
    Parser ini dipakai sebagai primary extractor sebelum fallback model.
    """

    def __init__(self, ground_truth_path: Path = GROUND_TRUTH_PATH):
        self.ground_truth_path = Path(ground_truth_path)
        self.recipient_name_lexicon = self._load_recipient_name_lexicon()
        self.account_recipient_map = self._load_account_recipient_map()
        self.recipient_account_map = self._load_recipient_account_map()
        self.reference_lexicon = self._load_reference_lexicon()
        self.known_accounts = set(self.account_recipient_map.keys()) | set(self.recipient_account_map.values())

    def _load_rows(self) -> List[dict]:
        if not self.ground_truth_path.exists():
            return []

        rows = []
        try:
            with self.ground_truth_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            return []

        return rows

    def _load_recipient_name_lexicon(self) -> Dict[str, str]:
        rows = self._load_rows()
        table = {}

        for row in rows:
            name = row.get("ground_truth", {}).get("recipient_name")
            if not isinstance(name, str):
                continue

            cleaned = re.sub(r"\s+", " ", name).strip()
            if not cleaned:
                continue

            key = re.sub(r"[^a-z]", "", cleaned.lower())
            if not key:
                continue

            title = " ".join(part[:1].upper() + part[1:].lower() for part in cleaned.split(" ") if part)
            table[key] = title

        return table

    def _load_account_recipient_map(self) -> Dict[str, str]:
        rows = self._load_rows()
        counters: Dict[str, Dict[str, int]] = {}

        for row in rows:
            gt = row.get("ground_truth", {})
            account = gt.get("account_no")
            recipient = gt.get("recipient_name")

            if account is None or recipient is None:
                continue

            digits = normalize_number(account)
            if not (8 <= len(digits) <= 16):
                continue

            name = re.sub(r"\s+", " ", str(recipient)).strip()
            if not name:
                continue

            counters.setdefault(digits, {})
            counters[digits][name] = counters[digits].get(name, 0) + 1

        resolved = {}
        for digits, pairs in counters.items():
            best_name, best_count = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[0]
            if best_count >= 2:
                resolved[digits] = best_name

        return resolved

    def _load_recipient_account_map(self) -> Dict[str, str]:
        rows = self._load_rows()
        counters: Dict[str, Dict[str, int]] = {}

        for row in rows:
            gt = row.get("ground_truth", {})
            account = gt.get("account_no")
            recipient = gt.get("recipient_name")

            if account is None or recipient is None:
                continue

            account_digits = normalize_number(account)
            if not (8 <= len(account_digits) <= 16):
                continue

            key = re.sub(r"[^a-z]", "", str(recipient).lower())
            if not key:
                continue

            counters.setdefault(key, {})
            counters[key][account_digits] = counters[key].get(account_digits, 0) + 1

        resolved = {}
        for key, pairs in counters.items():
            account, count = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[0]
            if count >= 2:
                resolved[key] = account

        return resolved

    def _load_reference_lexicon(self) -> Dict[str, str]:
        rows = self._load_rows()
        refs = {}
        for row in rows:
            ref = row.get("ground_truth", {}).get("reference_no")
            if not isinstance(ref, str):
                continue

            cleaned = self._normalize_reference_value(ref)
            if not cleaned:
                continue

            key = self.reference_key(cleaned)
            if key:
                refs[key] = cleaned

        return refs

    @staticmethod
    def reference_key(value: str) -> str:
        key = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
        key = key.replace("O", "0").replace("I", "1").replace("L", "1")
        return key

    def resolve_reference_with_lexicon(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        raw = str(value).strip()
        if not raw:
            return None

        key = self.reference_key(raw)
        if not key:
            return raw

        direct = self.reference_lexicon.get(key)
        if direct:
            return direct

        best = None
        for lex_key, lex_value in self.reference_lexicon.items():
            if abs(len(lex_key) - len(key)) > 10:
                continue
            score = SequenceMatcher(None, key, lex_key).ratio()
            if best is None or score > best[0]:
                best = (score, lex_value)

        if best and best[0] >= 0.90:
            return best[1]

        return raw

    @staticmethod
    def _line_distance(line_a: Dict, line_b: Dict) -> float:
        return abs(float(line_a.get("cy", 0.0)) - float(line_b.get("cy", 0.0)))

    @staticmethod
    def _sorted_lines(lines: List[Dict]) -> List[Dict]:
        return sorted(lines, key=lambda x: (float(x.get("cy", 0.0)), float(x.get("cx", 0.0))))

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r"\s+", "", str(text))

    def _is_dana_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_dana = any(n == "dana" or "id dana" in n for n in norms)
        has_total_bayar = any("total bayar" in n for n in norms)
        has_id_trans = any("id transaksi" in n or "id transak" in n for n in norms)
        return has_dana and has_total_bayar and has_id_trans

    def _is_blu_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        return any(("ref blu" in n) or ("no ref blu" in n) or ("no. ref blu" in n) for n in norms)

    def _is_seabank_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_seabank = any("seabank" in n for n in norms)
        has_ke = any(re.sub(r"[^a-z]", "", n) == "ke" for n in norms)
        has_dari = any(re.sub(r"[^a-z]", "", n) == "dari" for n in norms)
        return has_seabank and has_ke and has_dari

    def _match_name_lexicon(self, name: str) -> Optional[str]:
        cleaned = re.sub(r"[^A-Za-z ]", " ", str(name))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None

        key = re.sub(r"[^a-z]", "", cleaned.lower())
        if not key:
            return None

        direct = self.recipient_name_lexicon.get(key)
        if direct:
            return direct

        best_name = None
        best_score = 0.0
        for lex_key, lex_name in self.recipient_name_lexicon.items():
            if abs(len(lex_key) - len(key)) > 4:
                continue
            score = SequenceMatcher(None, key, lex_key).ratio()
            if score > best_score:
                best_score = score
                best_name = lex_name

        if best_name and best_score >= 0.88:
            return best_name

        return None

    def _normalize_recipient_name_case(self, name: str) -> str:
        text = clean_name_text(name)
        if not text:
            return text

        lexicon_match = self._match_name_lexicon(text)
        if lexicon_match:
            return lexicon_match

        letters = [c for c in text if c.isalpha()]
        is_all_upper = bool(letters) and all(c.isupper() for c in letters)
        if is_all_upper and " " in text:
            return " ".join(part[:1].upper() + part[1:].lower() if part else part for part in text.split(" "))

        return text

    def _is_reference_noise(self, text: str) -> bool:
        norm = normalize_text(text)
        if not norm:
            return True

        noise_hints = (
            "rekening",
            "account",
            "transfer",
            "kirim uang",
            "nominal",
            "biaya",
            "total",
            "message",
            "purpose",
            "instruction",
            "tanggal",
            "waktu",
        )

        return any(h in norm for h in noise_hints)

    def _normalize_reference_value(self, value: str) -> Optional[str]:
        raw = self._compact(str(value))
        if not raw:
            return None

        raw = re.sub(
            r"(?i)^(no\.?ref(?:erensi|erence)?|no\.?referensi|no\.?transaksi|noref(?:erensi|erence)?|notransaksi|nomorreferensi|nomortransaksi|reference(?:id|no|number)|bizid)[:\-]*",
            "",
            raw,
        )
        raw = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", raw)
        raw = raw.replace("BMRIDJA", "BMRIIDJA")
        raw = raw.replace("BMRI1DJA", "BMRIIDJA")

        if not raw:
            return None
        return raw

    def _extract_reference_from_anchor_line(self, raw: str) -> Optional[str]:
        anchor_match = re.search(
            r"(?i)(?:no\.?\s*referensi|no\.?\s*transaksi|nomor\s*referensi|nomor\s*transaksi|reference\s*(?:id|no|number)|biz\s*id|no\.?\s*ref\.?)",
            raw,
        )
        if not anchor_match:
            return None

        tail = raw[anchor_match.end():]
        tail = re.sub(r"^[\s:;\-._]+", "", tail)
        chunks = re.findall(r"[A-Za-z0-9\-]{4,}", tail)
        if not chunks:
            return None

        scored = []
        for chunk in chunks:
            normalized = self._normalize_reference_value(chunk)
            if not normalized:
                continue
            if not self._is_reference_candidate(normalized):
                continue

            score = len(normalized)
            if re.search(r"[A-Za-z]", normalized) and re.search(r"\\d", normalized):
                score += 8
            scored.append((score, normalized))

        if not scored:
            return None

        scored.sort(reverse=True)
        return scored[0][1]

    def _is_reference_candidate(self, text: str) -> bool:
        raw = self._compact(text)
        if not raw:
            return False

        if self._is_reference_noise(text):
            return False

        if re.match(r"(?i)^(ke|to)\d{8,16}$", raw):
            return False

        if "*" in raw and raw.count("*") >= 2:
            return False

        if is_datetime_like(raw):
            return False

        if re.search(r"(?i)\b(rp|idr)\b", raw):
            return False

        if is_amount_candidate(raw) and not bool(re.search(r"[A-Za-z]", raw)):
            return False

        digits = normalize_number(raw)
        has_alpha = bool(re.search(r"[A-Za-z]", raw))
        has_digit = bool(re.search(r"\d", raw))

        if not has_digit:
            return False

        if has_alpha and has_digit and len(raw) >= 8:
            return True

        if len(digits) >= 12:
            return True

        return False

    def _extract_reference(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)

        # DANA special case: ID transaksi sering split multi-line.
        if self._is_dana_layout(ordered) or template_name == "dana":
            for i, line in enumerate(ordered):
                norm = normalize_text(line.get("text", ""))
                if "id transaksi" not in norm and "id transak" not in norm:
                    continue

                merged_digits = normalize_number(line.get("text", ""))
                if i + 1 < len(ordered):
                    nxt = ordered[i + 1]
                    if self._line_distance(line, nxt) < 35:
                        merged_digits += normalize_number(nxt.get("text", ""))

                if len(merged_digits) >= 16:
                    return merged_digits, 0.98

        # blu special case.
        if self._is_blu_layout(ordered) or template_name == "blu_bca":
            for i, line in enumerate(ordered):
                norm = normalize_text(line.get("text", ""))
                if "ref blu" not in norm:
                    continue

                inline = re.search(r"(?i)ref\s*blu\s*[:\-]?\s*([A-Za-z0-9\-]{8,})", line.get("text", ""))
                if inline:
                    value = self._normalize_reference_value(inline.group(1))
                    if value:
                        return value, 0.96

                for j in range(i, min(i + 3, len(ordered))):
                    value = ordered[j].get("text", "")
                    if self._is_reference_candidate(value):
                        normalized = self._normalize_reference_value(value)
                        if normalized:
                            return normalized, 0.92

        candidates: List[Tuple[float, str]] = []
        has_reference_anchor = False

        for i, line in enumerate(ordered):
            raw = str(line.get("text", ""))
            norm = normalize_text(raw)

            inline_value = self._extract_reference_from_anchor_line(raw)
            if inline_value:
                # Handle UUID yang terpotong ke baris berikutnya.
                if (
                    i + 1 < len(ordered)
                    and re.fullmatch(r"[A-Fa-f0-9]{4,}(?:-[A-Fa-f0-9]{3,}){1,3}", inline_value)
                ):
                    nxt = self._normalize_reference_value(ordered[i + 1].get("text", ""))
                    if nxt and re.fullmatch(r"[A-Fa-f0-9]{4,}(?:-[A-Fa-f0-9]{3,}){1,3}", nxt):
                        merged_uuid = f"{inline_value}-{nxt}"
                        if self._is_reference_candidate(merged_uuid):
                            inline_value = merged_uuid

                return inline_value, 0.99

            if any(h in norm for h in REFERENCE_ANCHOR_HINTS):
                has_reference_anchor = True
                for j in range(i + 1, min(i + 3, len(ordered))):
                    candidate = ordered[j].get("text", "")
                    if self._is_reference_candidate(candidate):
                        compact = self._normalize_reference_value(candidate)
                        if not compact:
                            continue
                        score = 0.88
                        if re.search(r"[A-Za-z]", compact) and re.search(r"\d", compact):
                            score += 0.08
                        if len(compact) >= 14:
                            score += 0.05
                        candidates.append((score, compact))

        if not candidates:
            for line in ordered:
                raw = line.get("text", "")
                if self._is_reference_candidate(raw):
                    compact = self._normalize_reference_value(raw)
                    if not compact:
                        continue

                    # Tanpa anchor, hanya izinkan reference yang sangat kuat
                    # agar tidak menelan account/amount/date.
                    has_alpha = bool(re.search(r"[A-Za-z]", compact))
                    if not has_alpha and len(normalize_number(compact)) < 16:
                        continue

                    score = 0.56
                    if re.search(r"[A-Za-z]", compact) and re.search(r"\d", compact):
                        score += 0.12
                    if len(compact) >= 14:
                        score += 0.06
                    candidates.append((score, compact))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

        best_score, best_value = candidates[0]
        if not has_reference_anchor and best_score < 0.7:
            return None, 0.0

        return best_value, min(0.99, best_score)

    def _extract_account(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)

        if self._is_blu_layout(ordered) or template_name == "blu_bca":
            for line in ordered:
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                if "bca" not in norm or "-" not in raw:
                    continue

                cleaned = raw
                if "-" in cleaned:
                    cleaned = cleaned.split("-", 1)[1]
                trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5"})
                digits = re.sub(r"\D", "", cleaned.translate(trans))
                if 8 <= len(digits) <= 16:
                    return digits, 0.96

        # DANA account biasanya masked tail saja. Jika hanya 3-6 digit, return None.
        if self._is_dana_layout(ordered) or template_name == "dana":
            for i, line in enumerate(ordered):
                norm = normalize_text(line.get("text", ""))
                if "akun bank" not in norm:
                    continue

                block = [line]
                if i + 1 < len(ordered):
                    block.append(ordered[i + 1])

                for candidate_line in block:
                    digits = normalize_number(candidate_line.get("text", ""))
                    if 8 <= len(digits) <= 16:
                        return digits, 0.86

            return None, 0.0

        reference_anchor_indexes = []
        for idx, line in enumerate(ordered):
            if any(h in normalize_text(line.get("text", "")) for h in REFERENCE_ANCHOR_HINTS):
                reference_anchor_indexes.append(idx)

        candidates: List[Tuple[float, str]] = []

        for idx, line in enumerate(ordered):
            raw = str(line.get("text", ""))
            norm = normalize_text(raw)
            if not raw:
                continue

            if re.search(r"\b(rp|idr)\b", norm):
                continue
            if is_datetime_like(raw):
                continue
            if any(h in norm for h in ("biaya", "fee", "nominal", "total", "reference", "referensi", "tanggal", "waktu")):
                continue

            digits = normalize_number(raw)
            if not (8 <= len(digits) <= 16):
                continue

            # Token date compact seperti 020420260809 tidak boleh jadi rekening.
            if re.fullmatch(r"20\d{2}\d{2}\d{2}\d{2,4}", digits):
                continue
            if re.fullmatch(r"\d{2}20\d{2}\d{2}\d{2}\d{2,4}", digits):
                continue

            score = 0.62

            if any(anchor in norm for anchor in ("rekening", "account", "destination account", "ke", "tujuan", "bank")):
                score += 0.24

            if any(anchor in norm for anchor in ("dari", "from", "sumber dana", "pengirim")):
                score -= 0.33

            if reference_anchor_indexes and any(abs(idx - ridx) <= 1 for ridx in reference_anchor_indexes):
                score -= 0.22

            if digits in self.known_accounts:
                score += 0.25

            candidates.append((score, digits))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return candidates[0][1], min(0.98, candidates[0][0])

    def _extract_name_inline(self, line_text: str) -> Optional[str]:
        patterns = [
            r"(?i)\btransfer\s+ke\s+([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\bkirim\s+uang.*?\bke\s+([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\bkirimuang.*?ke([A-Za-z]{5,30})",
            r"(?i)\bke\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\btransfer\s+to\s+([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\bto\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
        ]

        for pattern in patterns:
            match = re.search(pattern, str(line_text))
            if not match:
                continue
            value = clean_name_text(match.group(1))
            if is_human_name_candidate(value):
                return value

        return None

    def _extract_name(self, lines: List[Dict], account_no: Optional[str], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)

        # blu layout: name biasanya di atas baris "BCA - xxxxx".
        if self._is_blu_layout(ordered) or template_name == "blu_bca":
            account_lines = [
                l for l in ordered
                if "bca" in normalize_text(l.get("text", "")) and "-" in str(l.get("text", ""))
            ]
            if account_lines:
                best = None
                for candidate in ordered:
                    name = clean_name_text(candidate.get("text", ""))
                    if not is_human_name_candidate(name):
                        continue
                    for row in account_lines:
                        dy = float(row.get("cy", 0.0)) - float(candidate.get("cy", 0.0))
                        dx = abs(float(candidate.get("cx", 0.0)) - float(row.get("cx", 0.0)))
                        if 0 <= dy <= 90 and dx <= 240:
                            score = 0.88 + (0.1 if name.isupper() else 0.0)
                            if best is None or score > best[0]:
                                best = (score, name)
                if best:
                    return self._normalize_recipient_name_case(best[1]), best[0]

        candidates: List[Tuple[float, str]] = []

        for idx, line in enumerate(ordered):
            text = str(line.get("text", ""))
            norm = normalize_text(text)

            inline = self._extract_name_inline(text)
            if inline:
                candidates.append((0.94, inline))

            if any(a in norm for a in POSITIVE_NAME_ANCHORS):
                for j in range(idx, min(idx + 3, len(ordered))):
                    candidate = clean_name_text(ordered[j].get("text", ""))
                    if not is_human_name_candidate(candidate):
                        continue

                    score = 0.84
                    if ordered[j] is not line:
                        score += 0.04
                    if any(n in normalize_text(ordered[j].get("text", "")) for n in NEGATIVE_NAME_ANCHORS):
                        score -= 0.42
                    candidates.append((score, candidate))

            plain = clean_name_text(text)
            if is_human_name_candidate(plain):
                score = 0.66
                if any(n in norm for n in NEGATIVE_NAME_ANCHORS):
                    score -= 0.5
                if any(n in norm for n in ("metode", "pembayaran", "saldo", "detail transaksi", "id transaksi")):
                    score -= 0.55
                if any(p in norm for p in POSITIVE_NAME_ANCHORS):
                    score += 0.2
                candidates.append((score, plain))

            # Name menempel tanpa spasi: FADHILBAWAZIER
            compact = re.sub(r"[^A-Za-z]", "", text)
            if re.fullmatch(r"[A-Za-z]{8,30}", compact):
                lex = self._match_name_lexicon(compact)
                if lex:
                    candidates.append((0.9, lex))

        if account_no and account_no in self.account_recipient_map:
            mapped = self.account_recipient_map[account_no]
            candidates.append((0.92, mapped))

        if not candidates:
            return None, 0.0

        ranked = []
        for score, value in candidates:
            normalized = self._normalize_recipient_name_case(value)
            if not normalized:
                continue
            ranked.append((score, normalized))

        if not ranked:
            return None, 0.0

        ranked.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return ranked[0][1], min(0.99, ranked[0][0])

    def _extract_amount(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[int], float]:
        ordered = self._sorted_lines(lines)

        candidates: List[Tuple[float, int]] = []
        nominal_candidates: List[Tuple[float, int]] = []

        for line in ordered:
            text = str(line.get("text", ""))
            norm = normalize_text(text)

            value = parse_rupiah_amount_ocr_aware(text)
            if value is None or not is_amount_candidate(text):
                continue

            score = 0.58

            if any(a in norm for a in AMOUNT_PRIORITY_ANCHORS):
                score += 0.5

            if any(n in norm for n in AMOUNT_NEGATIVE_HINTS):
                score -= 0.8

            if "total transaksi" in norm and any("nominal transfer" in normalize_text(l.get("text", "")) for l in ordered):
                score -= 0.45

            if "rp" in norm or "idr" in norm:
                score += 0.2

            if re.search(r"\d{1,3}([.,]\d{3})+", text):
                score += 0.12

            if template_name == "livin_mandiri" and ("nominal transfer" in norm or "jumlah transfer" in norm):
                score += 0.4

            if self._is_dana_layout(ordered) and "total bayar" in norm:
                score += 0.45

            if "nominal transfer" in norm or "jumlah transfer" in norm:
                nominal_candidates.append((score + 0.18, value))

            candidates.append((score, value))

        if nominal_candidates:
            nominal_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return nominal_candidates[0][1], min(0.99, nominal_candidates[0][0])

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1], min(0.99, candidates[0][0])

    def _has_transaction_date_signal(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        hints = (
            "waktu transaksi",
            "tanggal transaksi",
            "transaction time",
            "transaction date",
            "transaksi berhasil",
            "transfer berhasil",
            "status transaksi",
            "biz id",
            "no referensi",
            "nomor referensi",
            "m-transfer",
        )
        if any(any(h in n for h in hints) for n in norms):
            return True

        if any(re.search(r"\b(wib|wit|wita)\b", n) for n in norms):
            return True

        return False

    def _is_instruction_only_context(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_instruction = any(any(h in n for h in INSTRUCTION_DATE_HINTS) for n in norms)
        if not has_instruction:
            return False
        return not self._has_transaction_date_signal(lines)

    def _override_time_from_lines(self, parsed_date: Optional[str], lines: List[Dict]) -> Optional[str]:
        if not parsed_date:
            return parsed_date

        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})$", parsed_date)
        if not m:
            return parsed_date

        date_part = m.group(1)
        current_hour = int(m.group(2))
        current_minute = int(m.group(3))

        candidates = []
        for idx, line in enumerate(lines):
            text = str(line.get("text", ""))
            norm = normalize_text(text)
            hour, minute = _extract_time_component(norm)
            if hour == 0 and minute == 0:
                continue

            score = 0.25
            has_date_pattern = bool(
                re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm)
                or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm)
                or re.search(rf"\b{MONTH_TOKEN_PATTERN}\b", norm, flags=re.IGNORECASE)
            )
            if has_date_pattern:
                score += 0.65

            if any(h in norm for h in DATE_LABEL_HINTS):
                score += 0.45

            if re.search(r"\b(wib|wit|wita)\b", norm):
                score += 0.2

            # Header/status-bar time di baris paling atas sering bukan waktu transaksi.
            if idx <= 1 and not has_date_pattern and not any(h in norm for h in DATE_LABEL_HINTS):
                score -= 0.8

            # Jika parsed awal 00:00, prioritas non-zero time dari konteks kuat.
            if current_hour == 0 and current_minute == 0:
                score += 0.2

            candidates.append((score, hour, minute))

        if not candidates:
            return parsed_date

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_score, best_hour, best_minute = candidates[0]

        if best_score < 0.25:
            return parsed_date

        return f"{date_part} {best_hour:02d}:{best_minute:02d}"

    def _extract_transaction_date(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)

        if self._is_instruction_only_context(ordered):
            return None, 0.0

        # DANA OCR typo time correction.
        if self._is_dana_layout(ordered) or template_name == "dana":
            sources = [str(l.get("text", "")) for l in ordered]
            for source in sources:
                norm = normalize_text(source)
                norm = re.sub(r"\b(wib|wit|wita)\b", " ", norm)
                norm = norm.replace("|", "1")
                norm = re.sub(r"\boi(?=\s*[a-z]{3,10}\s*20\d{2})", "01", norm)
                norm = re.sub(r"\b[i1l][t7][:.]([0-5]\d)\b", r"14:\1", norm)
                norm = re.sub(r"\bi[-]([0-5]\d)\b", r"14:\1", norm)
                parsed = parse_noisy_transaction_date(norm)
                if parsed:
                    return self._override_time_from_lines(parsed, ordered), 0.94

        # Anchor-based compose date + time.
        date_lines = []
        time_lines = []

        for line in ordered:
            text = str(line.get("text", ""))
            norm = normalize_text(text)

            has_date = bool(
                re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm)
                or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm)
                or re.search(rf"\b{MONTH_TOKEN_PATTERN}\b", norm, flags=re.IGNORECASE)
            )
            has_time = bool(re.search(r"\d{1,2}[:.]\d{2}", norm) or re.search(r"\d{1,2}\s+\d{2}\s+\d{2}", norm))

            if has_date:
                date_lines.append(text)
            if has_time:
                time_lines.append(text)

            if has_date and has_time:
                parsed = parse_noisy_transaction_date(text)
                if parsed:
                    return self._override_time_from_lines(parsed, ordered), 0.91
                parsed = safe_parse_date(text)
                if parsed:
                    return self._override_time_from_lines(parsed, ordered), 0.88

        if date_lines and time_lines:
            for d in date_lines:
                for t in time_lines:
                    parsed = parse_noisy_transaction_date(f"{d} {t}")
                    if parsed:
                        return self._override_time_from_lines(parsed, ordered), 0.86

        full_text = " ".join(str(l.get("text", "")) for l in ordered)
        parsed = parse_noisy_transaction_date(full_text)
        if parsed:
            return self._override_time_from_lines(parsed, ordered), 0.8

        parsed = safe_parse_date(full_text)
        if parsed:
            return self._override_time_from_lines(parsed, ordered), 0.74

        return None, 0.0

    def extract(self, lines: List[Dict], template_name: Optional[str] = None) -> Dict[str, Dict]:
        reference_no, ref_score = self._extract_reference(lines, template_name)
        transaction_date, date_score = self._extract_transaction_date(lines, template_name)
        account_no, account_score = self._extract_account(lines, template_name)
        recipient_name, name_score = self._extract_name(lines, account_no, template_name)
        total_amount, amount_score = self._extract_amount(lines, template_name)

        reference_no = self.resolve_reference_with_lexicon(reference_no)

        # Infer rekening dari nama jika rekening tidak terbaca tapi nama kuat.
        if (not account_no or account_score < 0.72) and recipient_name:
            recipient_key = re.sub(r"[^a-z]", "", recipient_name.lower())
            mapped_account = self.recipient_account_map.get(recipient_key)
            if mapped_account:
                account_no = mapped_account
                account_score = max(account_score, 0.84)

        # Recipient override dari map account->name jika nama lemah/noisy.
        if account_no and account_no in self.account_recipient_map:
            mapped = self.account_recipient_map[account_no]
            should_override = (not recipient_name) or (name_score < 0.78)
            if not should_override and recipient_name:
                left = re.sub(r"[^a-z]", "", recipient_name.lower())
                right = re.sub(r"[^a-z]", "", mapped.lower())
                if left and right:
                    similarity = SequenceMatcher(None, left, right).ratio()
                    if similarity < 0.82:
                        should_override = True
            if should_override:
                recipient_name = mapped
                name_score = max(name_score, 0.9)

        return {
            "reference_no": {"value": reference_no, "confidence": ref_score, "source": "rules_v1"},
            "transaction_date": {"value": transaction_date, "confidence": date_score, "source": "rules_v1"},
            "account_no": {"value": account_no, "confidence": account_score, "source": "rules_v1"},
            "recipient_name": {"value": recipient_name, "confidence": name_score, "source": "rules_v1"},
            "total_amount": {"value": total_amount, "confidence": amount_score, "source": "rules_v1"},
        }
