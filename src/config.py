from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class EnvironmentConfig:
    num_workers: int

    data_dir: str
    working_dir: str
    output_dir: str
    checkpoint_dir: str
    submission_dir: str


@dataclass
class PreprocesssConfig:
    process_cqf: bool
    ref_voltage: float
    apply_filter: bool
    cutoff_freqs: tuple[float, float]


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
    agg_policy: str
    duration: int
    stride: int
    seed: int
    population_threshold: float


@dataclass
class WandbConfig:
    project: str
    mode: str  # offline, online, disabled


@dataclass
class PseudoLabelConfig:
    enabled: bool
    teacher_ensemble_name: str
    max_drop_ratio: float
    saturated_epochs: int


@dataclass
class LabelConfig:
    diversity_power: float
    population_power: float
    max_votes: int


@dataclass
class LrAdjustmentConfig:
    pattern: str
    ratio: float


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    lr_adjustments: list[LrAdjustmentConfig]
    weight_decay: float
    batch_size: int
    save_last: bool
    save_best: bool
    duration: int
    num_samples_per_eeg: int
    log_file_name: str
    random_seed_offset: int
    population_threshold: float
    no_decay_bias_params: bool
    transform: DictConfig | None
    train_dataset: DictConfig
    valid_dataset: DictConfig
    data: DataConfig
    optimizer: DictConfig
    scheduler: SchedulerConfig
    val: EvalConfig
    pseudo_label: PseudoLabelConfig
    label: LabelConfig


@dataclass
class ArchitectureConfig:
    in_channels: int
    out_channels: int
    recover_dual: bool
    use_lr_feature: bool
    use_similarity_feature: bool
    hidden_dim: int
    input_mask: bool
    use_bg_spec: bool
    lr_mapping_type: str
    spec_cropped_duration: int
    bg_spec_mask_value: float
    model_class: DictConfig
    model_checker: Callable[..., None]
    model: DictConfig


@dataclass
class InferConfig:
    batch_size: int
    num_samples: int
    model_choice: str
    log_name: str
    tta_iterations: int
    tta: DictConfig | None
    test_dataset: DictConfig


@dataclass
class MainConfig:
    job_name: str
    exp_name: str
    description: str
    phase: str
    fold: int
    seed: int
    debug: bool
    dry_run: bool
    cleanup: bool
    final_submission: bool
    check_only: bool
    preprocess: PreprocesssConfig
    env: EnvironmentConfig
    split: SplitConfig
    architecture: ArchitectureConfig
    trainer: TrainerConfig
    infer: InferConfig
    wandb: WandbConfig


@dataclass
class EnsembleFoldConfig:
    split: int
    seeds: list[int]


@dataclass
class EnsembleExperimentConfig:
    exp_name: str
    ensemble_seeds: list[int]
    folds: list[EnsembleFoldConfig]


@dataclass
class EnsembleEntityConfig:
    name: str
    experiments: list[EnsembleExperimentConfig]


@dataclass
class EnsembleMainConfig:
    ensemble_entity: EnsembleEntityConfig
    phase: str
    dry_run: bool
    debug: bool
    cleanup: bool
    final_submission: bool
    env: EnvironmentConfig
