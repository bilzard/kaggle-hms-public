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
class EvalConfig:
    batch_size: int
    aggregation_fn: str
    duration: int
    stride: int


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    batch_size: int
    save_last: bool
    save_best: bool
    duration: int
    data: DataConfig
    optimizer: DictConfig
    scheduler: SchedulerConfig
    val: EvalConfig


@dataclass
class ArchitectureConfig:
    in_channels: int
    out_channels: int
    model: DictConfig


@dataclass
class MainConfig:
    job_name: str
    exp_name: str
    phase: str
    fold: int
    seed: int
    debug: bool
    dry_run: bool
    cleanup: bool
    preprocess: PreprocesssConfig
    env: EnvironmentConfig
    split: SplitConfig
    architecture: ArchitectureConfig
    trainer: TrainerConfig
