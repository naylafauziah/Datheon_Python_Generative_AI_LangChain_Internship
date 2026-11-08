"""Strict Pydantic V2 schemas shared with the LLM structured-output task."""
import re
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class InvoiceItem(StrictSchema):
    description: str = Field(min_length=1)
    quantity: float | None = Field(default=None, ge=0)
    unit_price: float | None = Field(default=None, ge=0)
    line_total: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def line_amount_consistent(self):
        if all(v is not None for v in (self.quantity, self.unit_price, self.line_total)):
            if abs(self.quantity * self.unit_price - self.line_total) > 0.05:
                raise ValueError('quantity * unit_price tidak sama dengan line_total')
        return self


class Invoice(StrictSchema):
    document_type: Literal['invoice'] = 'invoice'
    vendor: str = Field(min_length=1)
    invoice_number: str | None = None
    invoice_date: date | None = None
    currency: str | None = Field(default=None, description='ISO code such as IDR or USD')
    line_items: list[InvoiceItem] = Field(default_factory=list)
    subtotal: float | None = Field(default=None, ge=0)
    tax: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)

    @field_validator('currency')
    @classmethod
    def check_currency(cls, value):
        if value is not None and not re.fullmatch(r'[A-Z]{3}', value):
            raise ValueError('currency harus berupa kode ISO 3 huruf seperti IDR')
        return value

    @model_validator(mode='after')
    def totals_consistent(self):
        if all(v is not None for v in (self.subtotal, self.tax, self.total)):
            # Allow the source document to round invoice totals by a few cents.
            if abs(self.subtotal + self.tax - self.total) > 0.05:
                raise ValueError('subtotal + tax tidak sama dengan total di dokumen')
        return self


class Experience(StrictSchema):
    company: str = Field(min_length=1)
    position: str | None = None
    period: str | None = None


class Education(StrictSchema):
    institution: str = Field(min_length=1)
    degree: str | None = None
    period: str | None = None


class Resume(StrictSchema):
    document_type: Literal['resume'] = 'resume'
    name: str = Field(min_length=1)
    email: str | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @field_validator('email')
    @classmethod
    def check_email(cls, value):
        if value is not None and not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
            raise ValueError('Format email tidak valid')
        return value
