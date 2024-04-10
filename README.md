# About This Repository

This repository contains resources to reproduce submissions for the competition [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)

## Setup

### Install All Dependencies for Development

```bash
pip install --editable .
```

## How to Reproduce

### Pre-Process

```bash
python -m run.preprocess job_name=preprocess phase=train
```

### Fold-split

```bash
python -m run.fold_split job_name=fold_split phase=train
```

### Train

#### Single experiment

```bash
python -m run.train --config-name=v5_eeg_24ep_cutmix job_name=train fold=0 seed=0
```

#### Batch execution

```bash
python schedule.py train --config_names=v5_eeg_24ep_cutmix --folds=0,1,2,3,4 --seeds=0,1,2
```

### Inference

#### Single experiment

```bash
python -m run.infer --config-name=v5_eeg_24ep_cutmix job_name=infer fold=0 seed=0
```

#### Batch execution

```bash
python schedule.py infer --config_names=v5_eeg_24ep_cutmix --folds=0,1,2,3,4 --seeds=0,1,2
```

#### Ensemble

```bash
python -m run.ensemble job_name=ensemble ensemble_entity=f01234_s012 ensemble_entity.name=v5_eeg_24ep_cutmix
```

### Batch Inference (Infer & Ensemble)

```bash
python -m run.batch_infer job_name=ensemble ensemble_entity=f01234_s012 ensemble_entity.name=v5_eeg_24ep_cutmix
```

### Generate Pseudo Labels

#### per eeg_id

```bash
python -m run.pseudo_label job_name=pseudo_label ensemble_entity=f01234_s012 ensemble_entity.name=v5_eeg_24ep_cutmix
```

#### per label_id

```bash
python -m run.pseudo_label_per_label job_name=pseudo_label_per_label ensemble_entity=f01234_s012 ensemble_entity.name=v5_eeg_24ep_cutmix
```

### Upload Checkpoints

```bash
python -m run.model_checkpoint ensemble_entity=bilzard_v1
```

## Performance

|exp_name|CV(n_votes>=8.4)|
|--|--|
|v5_eeg_24ep_cutmix|0.2477|

## Acknowledgements

## Reference
