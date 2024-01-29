from dataclasses import dataclass


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
class MainConfig:
    job_name: str
    phase: str
    debug: bool
    dry_run: bool
    cleanup: bool
    preprocess: PreprocesssConfig
    env: EnvironmentConfig
    split: SplitConfig
