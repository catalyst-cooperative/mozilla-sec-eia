"""Define metadata for development database."""

import datetime
from enum import Enum

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


class Base(DeclarativeBase, MappedAsDataclass):
    """Base for collecting metadata.

    For more info see:
    https://docs.sqlalchemy.org/en/20/tutorial/metadata.html#using-orm-declarative-forms-to-define-table-metadata
    """

    pass


class Sec10kMetadata(Base):
    """Metadata describing archived SEC 10K filings."""

    __tablename__ = "sec10k_metadata"

    filename: Mapped[str] = mapped_column(String(75), primary_key=True)
    cik: Mapped[int]
    company_name: Mapped[str]
    form_type: Mapped[str]
    date_filed: Mapped[datetime.date]
    year_quarter: Mapped[str]
    exhibit_21_version: Mapped[str]


class Basic10kBlock(Enum):
    """Basic 10k data is seperated into various blocks of data."""

    COMPANY_DATA: str = "company data"
    FILING_VALUES: str = "filing values"
    BUSINESS_ADDRESS: str = "business address"
    MAIL_ADDRESS: str = "mail address"
    FORMER_COMPANY: str = "former company"


class Basic10k(Base):
    """Define schema for basic 10k company info table."""

    __tablename__ = "basic_10k"

    filename: Mapped[str] = mapped_column(
        String(75), ForeignKey("sec10k_metadata.filename"), primary_key=True
    )
    filer_count: Mapped[int] = mapped_column(primary_key=True)
    block: Mapped[Basic10kBlock] = mapped_column(primary_key=True)
    block_count: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(), primary_key=True)
    value: Mapped[str]


class Exhibit21(Base):
    """Define schema for extracted exhibit 21 data."""

    __tablename__ = "exhbit_21"

    filename: Mapped[str] = mapped_column(
        String(75), ForeignKey("sec10k_metadata.filename"), primary_key=True
    )
    subsidiary: Mapped[str]
    location_of_incorporation: Mapped[str]
    ownership_percentage: Mapped[float]
