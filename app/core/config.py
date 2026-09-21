"""
Application configuration management.

Loads all environment variables via pydantic-settings and exposes
a single `settings` singleton used throughout the application.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_NAME: str = "ANPR Backend"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development", description="development | staging | production")
    DEBUG: bool = False

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False

    # ------------------------------------------------------------------ #
    # MongoDB
    # ------------------------------------------------------------------ #
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "anpr_db"
    MONGO_MAX_POOL_SIZE: int = 10
    MONGO_MIN_POOL_SIZE: int = 1
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 5000

    # ------------------------------------------------------------------ #
    # Model Weights
    # ------------------------------------------------------------------ #
    WEIGHTS_DIR: Path = Path("weights")
    VEHICLE_MODEL_PATH: Path = Path("weights/vehicle_yolo.pt")
    PLATE_MODEL_PATH: Path = Path("weights/license_plate_detector.pt")

    # Vehicle detection confidence thresholds
    VEHICLE_CONF_THRESHOLD: float = 0.45
    VEHICLE_IOU_THRESHOLD: float = 0.45

    # Plate detection confidence thresholds
    PLATE_CONF_THRESHOLD: float = 0.40
    PLATE_IOU_THRESHOLD: float = 0.45

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    OCR_LANG: str = "en"
    OCR_USE_GPU: bool = False
    OCR_MIN_CONFIDENCE: float = 0.60   # below this → LOW_CONFIDENCE result

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    UPLOAD_DIR: Path = Path("uploads")
    PLATES_DIR: Path = Path("uploads/plates")
    VEHICLES_DIR: Path = Path("uploads/vehicles")
    MAX_UPLOAD_SIZE_MB: int = 20

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("logs")
    LOG_FILE: str = "anpr.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    # ------------------------------------------------------------------ #
    # CORS (future-proofing for any internal tooling)
    # ------------------------------------------------------------------ #
    CORS_ORIGINS: list[str] = ["*"]

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("APP_ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}")
        return v

    @field_validator("OCR_MIN_CONFIDENCE", "VEHICLE_CONF_THRESHOLD", "PLATE_CONF_THRESHOLD")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("Confidence thresholds must be between 0 and 1 (exclusive)")
        return v

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        dirs = [
            self.UPLOAD_DIR,
            self.PLATES_DIR,
            self.VEHICLES_DIR,
            self.LOG_DIR,
            self.WEIGHTS_DIR,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


# Module-level alias for convenient import
settings: Settings = get_settings()
