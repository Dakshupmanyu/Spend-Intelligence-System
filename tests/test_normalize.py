import math

from spend_engine.normalize import parse_number, parse_vat_rate, normalize_text
from spend_engine.canonical import extract_po_and_item


def test_parse_number_international_formats():
    assert parse_number("1,234.56") == 1234.56
    assert parse_number("1.234,56") == 1234.56
    assert parse_number("123,45") == 123.45


def test_vat_formats():
    assert math.isclose(parse_vat_rate("15%"), 0.15)
    assert math.isclose(parse_vat_rate("0.15"), 0.15)
    assert math.isclose(parse_vat_rate("15"), 0.15)
    assert math.isclose(parse_vat_rate(""), 0.15)


def test_po_reference_parser():
    assert extract_po_and_item("PO#4510000018/10.0") == ("4510000018", "10")
    assert extract_po_and_item("4510000018") == ("4510000018", "")


def test_normalize_text_removes_accents_and_punctuation():
    assert normalize_text("  MÁQUINA; PART-01 ") == "maquina part 01"


def test_numeric_identifier_removes_excel_zero_decimal_suffix():
    from spend_engine.normalize import normalize_identifier

    assert normalize_identifier("10.0", strip_leading_zeroes=True) == "10"
    assert normalize_identifier("00010.000", strip_leading_zeroes=True) == "10"


def test_parse_number_supports_accounting_negative_notation():
    assert parse_number("(1,234.56)") == -1234.56
    assert parse_number("123.45-") == -123.45
