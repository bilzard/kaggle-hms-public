# About This Repository

This repository contains resources to reproduce submissions for the competition [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)

## Setup

### Install All Dependencies for Development

```bash
pip install --editable .
```

## How to Reproduce

### Pre-process data

```bash
python -m run.preprocess job_name=preprocess phase=train,test
```

### Fold-split

```bash
python -m run.fold_split job_name=fold_split phase=train
```

### Train

```bash
python -m run.train --config-name=exp001 job_name=train fold=0 seed=42
```

## Result

## Acknowledgements

## Reference
