from datetime import date
import pytest
from pydantic import ValidationError
from src.schemas import Invoice, Resume


def test_valid_invoice():
    invoice = Invoice.model_validate({
        'vendor': 'PT Demo', 'invoice_number': 'A-100',
        'invoice_date': '2026-10-26', 'currency': 'IDR',
        'line_items': [{'description': 'Service', 'quantity': 2,
                        'unit_price': 100, 'line_total': 200}],
        'subtotal': 200, 'tax': 22, 'total': 222,
    })
    assert invoice.invoice_date == date(2026, 10, 26)
    assert invoice.total == 222
    assert Invoice.model_validate_json(invoice.model_dump_json()).total == 222


def test_invoice_total_mismatch_rejected():
    with pytest.raises(ValidationError, match='subtotal'):
        Invoice.model_validate({'vendor': 'PT Demo', 'subtotal': 200,
                                'tax': 22, 'total': 999})


def test_extra_field_rejected():
    with pytest.raises(ValidationError, match='Extra inputs'):
        Invoice.model_validate({'vendor': 'PT Demo', 'hidden': 'not allowed'})


def test_invoice_missing_optional_fields():
    invoice = Invoice.model_validate({'vendor': 'PT Demo'})
    assert invoice.invoice_number is None
    assert invoice.line_items == []


def test_bad_resume_email_rejected():
    with pytest.raises(ValidationError, match='email'):
        Resume.model_validate({'name': 'Nadia', 'email': 'not_an_email'})


def test_resume_valid():
    resume = Resume.model_validate({'name': 'Nadia', 'email': 'nadia@example.com',
                                    'skills': ['Python', 'SQL']})
    assert resume.skills == ['Python', 'SQL']


def test_line_item_math_rejected():
    with pytest.raises(ValidationError, match='quantity'):
        Invoice.model_validate({'vendor': 'PT Demo', 'line_items': [
            {'description': 'Service', 'quantity': 2, 'unit_price': 100, 'line_total': 100}
        ]})
