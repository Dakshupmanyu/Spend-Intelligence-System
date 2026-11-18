from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from datetime import datetime
from typing import Any

import pandas as pd
from dateutil import parser as date_parser


def text_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_text(value: Any) -> str:
    text = text_value(value).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_identifier(value: Any, strip_leading_zeroes: bool = False) -> str:
    text = text_value(value).upper()
    # CSV/Excel exports often render integer keys as e.g. "10.0". When
    # numeric normalization is requested, remove only a zero-only decimal
    # suffix before stripping punctuation, otherwise "10.0" becomes "100".
    if strip_leading_zeroes:
        text = re.sub(r"\.0+$", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    if strip_leading_zeroes and text.isdigit():
        text = text.lstrip("0") or "0"
    return text


def stable_key(prefix: str, namespace: str, identifier: str) -> str:
    raw = f"{namespace}|{identifier}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"{prefix}-{digest}"


def parse_number(value: Any, decimal: str = ".", thousands: str = ",") -> float | None:
    raw = text_value(value)
    if not raw:
        return None
    raw = raw.replace("\u00a0", " ").replace(" ", "")
    parenthesized_negative = raw.startswith("(") and raw.endswith(")")
    if parenthesized_negative:
        raw = raw[1:-1]
    trailing_negative = raw.endswith("-")
    if trailing_negative:
        raw = raw[:-1]
    raw = re.sub(r"[^0-9,\.\-+]", "", raw)
    if not raw or raw in {"-", "+", ".", ","}:
        return None

    # Detect common international forms when both separators occur.
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        tail = raw.rsplit(",", 1)[-1]
        if decimal == "," or len(tail) in {1, 2}:
            raw = raw.replace(thousands, "") if thousands and thousands != "," else raw
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif decimal and decimal != ".":
        raw = raw.replace(decimal, ".")
    try:
        number = float(raw)
        if parenthesized_negative or trailing_negative:
            number = -abs(number)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def parse_vat_rate(value: Any, default: float = 0.15) -> float:
    raw = text_value(value)
    if not raw:
        return default
    number = parse_number(raw, decimal=".", thousands=",")
    if number is None:
        return default
    if "%" in raw or number > 1:
        number /= 100.0
    if number < 0 or number > 1:
        return default
    return number


def parse_date(value: Any, formats: list[str] | None = None, dayfirst: bool = False) -> pd.Timestamp | pd.NaT:
    raw = text_value(value)
    if not raw:
        return pd.NaT
    for fmt in formats or []:
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except (ValueError, TypeError):
            continue
    # Excel serial dates are accepted only in the conventional 5-digit range.
    if re.fullmatch(r"\d{5}(\.0+)?", raw):
        try:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(raw), unit="D")
        except (ValueError, OverflowError):
            return pd.NaT
    try:
        return pd.Timestamp(date_parser.parse(raw, dayfirst=dayfirst, fuzzy=False))
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def month_key(value: Any) -> str:
    date = parse_date(value)
    return "" if pd.isna(date) else date.strftime("%Y-%m")


def token_similarity(left: Any, right: Any) -> float:
    def tokens(value: Any) -> set[str]:
        result = set()
        for token in normalize_text(value).split():
            # Conservative plural normalization helps common catalogue labels (fan/fans, ink/inks).
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            if token:
                result.add(token)
        return result

    a = tokens(left)
    b = tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity_ratio(left: Any, right: Any) -> float:
    try:
        from rapidfuzz.fuzz import ratio

        return float(ratio(normalize_text(left), normalize_text(right)))
    except ImportError:
        from difflib import SequenceMatcher

        return 100.0 * SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def normalize_supplier_name(value: Any) -> str:
    """Normalize spelling and remove common legal suffixes without erasing core tokens."""
    tokens = normalize_text(value).split()
    legal_suffixes = {
        "pty", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
        "gmbh", "sa", "plc", "bv", "ag", "co", "company", "sarl", "srl", "kg",
    }
    while tokens and tokens[-1] in legal_suffixes:
        tokens.pop()
    return " ".join(tokens)
