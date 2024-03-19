from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class EnvironmentConfig:
    name: str
    num_workers: int
    infer_batch_size: int
    grad_checkpointing: bool
    data_dir: str
    working_dir: str
    output_dir: str
    checkpoint_dir: str
    submission_dir: str


@dataclass
class CqfConfig:
    kernel_size: int
    top_k: int
    eps: float
    distance_threshold: float
    distance_metric: str
    normalize_type: str


@dataclass
class PreprocessConfig:
    process_cqf: bool
    clip_val: float
    ref_voltage: float
    apply_filter: bool
    cutoff_freqs: tuple[float | None, float | None]
    reject_freq: float | None
    drop_leftmost_nulls: bool
    pad_mode: str
    device: str
    cqf: CqfConfig


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
    weight_exponent: float
    min_weight: float


@dataclass
class WandbConfig:
    project: str
    mode: str  # offline, online, disabled


@dataclass
class ParamScheduleConfig:
    schedule_start_epoch: int
    target_epoch: int
    initial_value: float
    target_value: float


@dataclass
class WeightScheduleConfig:
    weight_exponent: ParamScheduleConfig
    min_weight: ParamScheduleConfig


@dataclass
class LabelConfig:
    diversity_power: float
    population_power: float
    max_votes: int
    label_postfix: list[str]
    weight_key: list[str]
    schedule: WeightScheduleConfig


@dataclass
class LrAdjustmentConfig:
    pattern: str
    ratio: float


@dataclass
class DistillationConfig:
    teacher_exp_name: str
    teacher_seed: int
    target_forget_rate: float
    target_epochs: int
    use_loss_weights: bool


@dataclass
class LossWeightConfig:
    norm_policy: str
    global_mean: float


@dataclass
class SslConfig:
    enabled: bool


@dataclass
class MeanTeacherConfig(SslConfig):
    average_type: str
    teacher_decay: float
    use_buffers: bool
    use_loss_weights: bool
    weight_schedule: ParamScheduleConfig
    decay_schedule: ParamScheduleConfig


@dataclass
class ContrastiveConfig(SslConfig):
    weight_schedule: ParamScheduleConfig


@dataclass
class TrainerConfig:
    trainer_class: DictConfig
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
    no_decay_bias_params: bool
    class_weights: list[float]
    class_weight_exponent: float
    use_loss_weights: bool
    transform: DictConfig | None
    train_dataset: DictConfig
    valid_dataset: DictConfig
    data: DataConfig
    optimizer: DictConfig
    scheduler: SchedulerConfig
    val: EvalConfig
    label: LabelConfig
    distillation: DistillationConfig
    loss_weight: LossWeightConfig
    ssl: type[SslConfig]


@dataclass
class ArchitectureConfig:
    in_channels: int
    in_channels_eeg: int
    in_channels_spec: int
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
class DevelopmentConfig:
    num_samples: int


@dataclass
class InferConfig:
    batch_size: int
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
    preprocess: PreprocessConfig
    env: EnvironmentConfig
    split: SplitConfig
    architecture: ArchitectureConfig
    trainer: TrainerConfig
    infer: InferConfig
    wandb: WandbConfig
    dev: DevelopmentConfig


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
    dev: DevelopmentConfig
