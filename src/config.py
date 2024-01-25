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
    cleanup: bool


@dataclass
class MainConfig:
    job_name: str
    phase: str
    debug: bool
    dry_run: bool
    preprocess: PreprocesssConfig
    env: EnvironmentConfig
