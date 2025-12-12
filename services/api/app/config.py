# services/api/app/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os, yaml

# --- Section models ---

class AppCfg(BaseModel):
    name: str = "ai-research-insights"
    env: str = "dev"
    log_level: str = "INFO"
    sse_heartbeat_seconds: int = 10
    allow_cors_origins: List[str] = []

class SecurityCfg(BaseModel):
    require_api_key: bool = True
    api_key: str = "dev"
    enable_hmac: bool = False
    rate_limit_per_min: int = 120

class TenancyCfg(BaseModel):
    strategy: str = "rls"  # rls | per_schema (reserved)
    default_tenant: str = "default"
    enforce: bool = True

class PgCfg(BaseModel):
    dsn: str

class OsCfg(BaseModel):
    endpoint: str
    index_prefix: str = "t_"
    use_vectors: bool = True
    vector_dim: int = 384
    bm25_k: int = 50
    vec_k: int = 50
    fusion_weight: float = 0.5  # 0..1 -> weight for vector vs bm25 in RRF/merge

class NeoCfg(BaseModel):
    uri: str
    user: str
    password: str

class MinioCfg(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str

class LLMCfg(BaseModel):
    base_url: str
    model: str
    max_input_tokens: int = 6000
    max_output_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 1.0

class RECfg(BaseModel):
    primary: str = "rebel"           # "rebel" | "openie6" | "corenlp"
    rebel_url: Optional[str] = None  # http://rebel-extractor:8000
    openie6_url: Optional[str] = None
    corenlp_url: Optional[str] = None
    min_confidence: float = 0.10

class GrobidCfg(BaseModel):
    url: str
    timeout_sec: int = 60

class PDFCfg(BaseModel):
    max_pages: int = 400
    max_size_mb: int = 50

class CSVThresholds(BaseModel):
    subject_ebio_min: float = 0.70
    subject_ngen_min: float = 0.70
    object_ebio_min: float = 0.70
    object_ngen_min: float = 0.70
    confidence_min: float = 0.10

class CSVCfg(BaseModel):
    thresholds: CSVThresholds = Field(default_factory=CSVThresholds)

class AVCfg(BaseModel):
    enabled: bool = False
    host: str = "clamav"
    port: int = 3310
    timeout_sec: int = 5

class StreamCfg(BaseModel):
    enable_sse: bool = True
    chunk_bytes: int = 2048

class ClassificationCfg(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str = "facebook/bart-large-mnli"
    labels: List[str] = ["has biomedical entity", "has generic noun", "other"]
    batch_size: int = 32
    max_concurrency: int = 8

class ExportCfg(BaseModel):
    triples_csv_bucket: str = "papers"
    triples_csv_object: str = "triplets_export/triplets.csv"

# --- Root settings ---

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    app: AppCfg
    security: SecurityCfg
    multi_tenancy: TenancyCfg
    postgres: PgCfg
    opensearch: OsCfg
    neo4j: NeoCfg
    minio: MinioCfg
    llm: LLMCfg
    re: RECfg
    grobid: GrobidCfg
    pdf: PDFCfg
    csv: CSVCfg
    antivirus: AVCfg = Field(default_factory=AVCfg)
    streaming: StreamCfg = Field(default_factory=StreamCfg)
    classification: ClassificationCfg = Field(default_factory=ClassificationCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        cfg_path = path or os.environ.get("APP_CONFIG", "/app/config/default.yaml")
        # 1) Read raw YAML
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = f.read()
        # 2) Expand ${VAR} using container environment
        raw = os.path.expandvars(raw)
        # 3) Parse YAML and construct settings
        data = yaml.safe_load(raw)
        return cls(**data)

# Single import point used by the whole app:
settings = Settings.load()

