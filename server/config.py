"""
Configuration for the beyondLDA2 API server.
All values are set via environment variables with sensible defaults.
"""

import os
from pathlib import Path


class Settings:
    # --- Server ---
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))

    # --- PostgreSQL ---
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "beyondlda2")
    DB_USER: str = os.getenv("DB_USER", "beyondlda2")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "beyondlda2")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    )
    DATABASE_URL_SYNC: str = os.getenv(
        "DATABASE_URL_SYNC",
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    )

    # --- Redis (Celery broker) ---
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL",
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    )
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND",
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    )

    # --- Worker ---
    WORK_DIR: str = os.getenv("WORK_DIR", "/tmp/beyondlda2-jobs")
    MPI_NPROCS: int = int(os.getenv("MPI_NPROCS", "4"))
    GPAW_SETUP_PATH: str | None = os.getenv("GPAW_SETUP_PATH")

    # --- ASE DB (scientific results) ---
    ASE_DB_DIR: str = os.getenv("ASE_DB_DIR", "/data/ase-dbs")


settings = Settings()
