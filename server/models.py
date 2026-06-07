"""
SQLAlchemy ORM models for the job queue (PostgreSQL).

Mirrors the ``simulations`` table from ruby_siesta — same schema pattern.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Simulation(Base):
    """One submitted GPAW calculation job."""

    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(
        String(20),
        nullable=False,
        default="queued",
        index=True,
    )
    # Valid: queued, processing, completed, failed

    payload = Column(JSON, nullable=False)
    """Full request parameters stored as JSON"""

    parsed_results = Column(JSON, nullable=True)
    """Structured results (energy, gap, etc.) — populated on completion"""

    fdf_output = Column(Text, nullable=True)
    """For compatibility with ruby_siesta naming; stores the generated
    Python script content that was/would-be executed."""

    error_log = Column(Text, nullable=True)
    """Exception traceback on failure"""

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
