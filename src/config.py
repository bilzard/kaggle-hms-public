from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class EnvironmentConfig:
    num_workers: int

    data_dir: str
    working_dir: str
    output_dir: str


@dataclass
class PreprocesssConfig:
    process_cqf: bool
    ref_voltage: float


@dataclass
class SplitConfig:
    strategy: str
    num_splits: int
    num_validation_patients: int


@dataclass
class SchedulerConfig:
    warmup_ratio: float


@dataclass
class DataConfig:
    target_key: str
    pred_key: str
    weight_key: str
    input_keys: list[str]


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    data: DataConfig
    optimizer: DictConfig
    scheduler: SchedulerConfig


@dataclass
class ArchitectureConfig:
    in_channels: int
    out_channels: int
    model: DictConfig


@dataclass
class MainConfig:
    job_name: str
    phase: str
    debug: bool
    dry_run: bool
    cleanup: bool
    preprocess: PreprocesssConfig
    env: EnvironmentConfig
    split: SplitConfig
    architecture: ArchitectureConfig
    trainer: TrainerConfig
