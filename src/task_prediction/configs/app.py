from importlib.metadata import version
from pathlib import Path
from pydantic import Field, BaseModel, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import LoggingConfig

class PredictorConfig(BaseModel):
    model_dir: Path
    alpha_smooth: float = Field(default=0.6, ge=0.0, le=1.0)
    force_stage_b: bool = True
    always_validate_input: bool = False

class DataConfig(BaseModel):
    short_sec: PositiveInt = 5
    mid_sec: PositiveInt = 10
    long_sec: PositiveInt = 25
    max_history_sec: PositiveInt = 27
    
    gaze_freq_hz: PositiveInt = 120
    min_availability_pct: float = Field(default=0.75, ge=0.0, le=1.0)
    min_validity_pct: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_window_durations(self) -> "DataConfig":
        if not (self.short_sec < self.mid_sec < self.long_sec < self.max_history_sec):
            raise ValueError('Window durations must be strictly increasing (short < mid < long < max_history).')
        return self

class NATSSinkConfig(BaseModel):
    enabled: bool = True
    subject: str = "intent.task_prediction"

class TerminalSinkConfig(BaseModel):
    enabled: bool = True
    refresh_per_sec: PositiveInt = 10

class ParquetSinkConfig(BaseModel):
    enabled: bool = True
    output_dir: Path = Path("./data")
    drop_when_full: bool = True
    max_buffer_size: PositiveInt = 5 # Flushes every 15 seconds for 3-second predictions
    queue_size: PositiveInt = 50

    @model_validator(mode='after')
    def validate_buffer_sizes(self) -> "ParquetSinkConfig":
        if self.queue_size <= self.max_buffer_size:
            raise ValueError('Queue must be bigger than buffer.')
        return self

class AppSettings(BaseSettings):
    model: PredictorConfig
    data: DataConfig = Field(default_factory=DataConfig)
    sampling_interval_ms: PositiveInt = 2_000

    # Used for sink and input data
    nats_host: str = "nats://localhost:4222"
    
    # Sinks
    parquet: ParquetSinkConfig = Field(default_factory=ParquetSinkConfig)
    nats: NATSSinkConfig = Field(default_factory=NATSSinkConfig)
    terminal: TerminalSinkConfig = Field(default_factory=TerminalSinkConfig)

    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    __version__: str = version("task-prediction")

    # Pydantic will look for e.g. INTENT__SAMPLING_INTERVAL_MS
    model_config = SettingsConfigDict(
        env_prefix="TASK_PRED__",
        env_file=".env",
        env_nested_delimiter="__", 
        case_sensitive=False,
    )

class OrchestratedSettings(AppSettings):
    data_dir: Path = Path("./data")
    health_subject: str = "intent.health.task_pred"
    cmds_subject: str = "intent.cmds.task_pred"