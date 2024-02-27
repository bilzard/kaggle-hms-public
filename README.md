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
python -m run.train --config-name=exp001 job_name=train fold=0 seed=42
```

#### Batch execution

```bash
python schedule.py train --config_names=exp001 --folds=0,1,2 --seeds=42,0,1
```

### Inference

#### Single experiment

```bash
python -m run.infer --config-name=exp001 job_name=infer fold=0 seed=42
```

#### Batch execution

```bash
python schedule.py infer --config_names=exp001 --folds=0,1,2 --seeds=42,0,1
```

## Result

## Acknowledgements

## Reference
