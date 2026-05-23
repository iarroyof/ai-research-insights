# services/api/app/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os, re, yaml


def _expand_env(raw: str) -> str:
    """
    Expand shell-like environment placeholders in YAML.

    os.path.expandvars does not understand ${VAR:-default}, which is useful in
    docker-compose configs. Keep ${VAR} behavior compatible with expandvars.
    """
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
        if default is not None:
            return default
        return match.group(0)

    return os.path.expandvars(pattern.sub(repl, raw))

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
    chat_provider: str = "local"  # local | nvidia
    max_input_tokens: int = 6000
    max_output_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 1.0
    context_manager_provider: str = "local"  # local | nvidia
    context_manager_model: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key: Optional[str] = None
    nvidia_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    nvidia_max_tokens: int = 2048
    nvidia_reasoning_effort: Optional[str] = None
    nvidia_enable_thinking: Optional[bool] = None

class MemoryCfg(BaseModel):
    enabled: bool = True
    working_buffer_turns: int = 8
    working_buffer_token_budget: int = 1800
    memory_k: int = 8
    triplet_k: int = 8
    web_k: int = 3
    token_budget_ratio: float = 0.45
    episodic_summary_turns: int = 6
    lifecycle_update_k: int = 80
    eviction_importance_threshold: float = 0.25
    allow_web_search_default: bool = False
    use_llm_reflection: bool = False
    shared_policy_enabled: bool = False
    reward_trace_enabled: bool = True
    auto_context_enabled: bool = True
    auto_context_k: int = 8
    auto_context_query_variants: int = 4
    auto_context_llm_refine: bool = True
    auto_context_llm_notes: bool = True
    contradiction_threshold: float = 0.35
    nli_enabled: bool = True
    nli_provider: str = "hf_api"  # hf_api | http | llm | heuristic | local
    nli_endpoint: Optional[str] = None
    nli_model: str = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    hf_api_token: Optional[str] = None
    hf_api_base_url: str = "https://router.huggingface.co/hf-inference/models"
    hf_api_timeout_sec: int = 45
    nli_hf_api_batch_size: int = 8
    nli_min_entailment: float = 0.55
    nli_contradiction_threshold: float = 0.45

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
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
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
        # 2) Expand ${VAR} and ${VAR:-default} using container environment
        raw = _expand_env(raw)
        # 3) Parse YAML and construct settings
        data = yaml.safe_load(raw)
        return cls(**data)

# Single import point used by the whole app:
settings = Settings.load()
