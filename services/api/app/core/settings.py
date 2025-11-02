# services/api/app/core/settings.py
from __future__ import annotations

import os
import pathlib
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict


class ExtractionProviderCfg(BaseModel):
    base_url: str


class ExtractionCfg(BaseModel):
    provider: str = "stanford"
    providers: Dict[str, ExtractionProviderCfg] = Field(default_factory=dict)


class SecurityCfg(BaseModel):
    # For local/dev, you can leave the key as "dev". In prod, set via env.
    require_api_key: bool = True
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", "dev"))


class AppCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")  # ignore unknown YAML keys
    extraction: ExtractionCfg = Field(default_factory=ExtractionCfg)
    security: SecurityCfg = Field(default_factory=SecurityCfg)


def _load_yaml_config() -> Dict[str, Any]:
    """
    Load YAML configuration from APP_CONFIG (if set) or /app/config/default.yaml.
    Missing file => return empty dict (defaults kick in).
    """
    cfg_path = os.getenv("APP_CONFIG", "/app/config/default.yaml")
    p = pathlib.Path(cfg_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


_cfg_dict = _load_yaml_config()
settings = AppCfg.model_validate(_cfg_dict)

