import json
import math
import shutil
from pathlib import Path
from typing import Any

import hydra
from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import EnsembleEntityConfig, ModelCheckpointMainConfig


def convert_size(size_bytes, num_dicimal_place=2):
    """
    バイト単位のサイズを人間が読みやすい形式に変換する関数
    """
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, num_dicimal_place)
    return f"{s} {size_names[i]}"


def copy_files_with_ensemble_entity(
    source_dir: Path,
    dest_dir: Path,
    cfg: EnsembleEntityConfig,
    checkpoint_name: str = "last_model.pth",
    dry_run: bool = False,
):
    """
    automatically copy files with the specified glob patterns in the source directory to the destination directory
    """
    source_paths = []
    total_size = 0
    for experiment in cfg.experiments:
        for fold in experiment.folds:
            for seed in fold.seeds:
                source_path = (
                    source_dir
                    / experiment.exp_name
                    / f"fold_{fold.split}"
                    / f"seed_{seed}"
                    / "model"
                    / checkpoint_name
                )
                if "debug" in source_path.parts:
                    continue
                source_paths.append(source_path)
                total_size += source_path.stat().st_size

    print(f"* Total files: {len(source_paths)}")
    print(f"* Total size: {convert_size(total_size)}")
    for source_path in source_paths:
        relative_path = source_path.relative_to(source_dir)
        dest_path = dest_dir / relative_path

        print(f"Try to copy {source_path} to {dest_path}...")
        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)

    # copy dummy data
    dummy_path = dest_dir / "dummy"
    print(f"Try to copy {dummy_path}...")
    if not dry_run:
        dummy_path.touch(exist_ok=True)


def update_dataset(user_name, title, tmp_dir, create: bool):
    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    api = KaggleApi()
    api.authenticate()

    if create:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )


@hydra.main(config_path="conf", config_name="model_checkpoint", version_base="1.2")
def main(cfg: ModelCheckpointMainConfig):
    if cfg.dry_run:
        print("=== dry run mode ===")
    tmp_dir = Path("./tmp")
    if not cfg.dry_run:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    working_dir = Path(cfg.env.working_dir)
    src_dir = working_dir / "train"

    copy_files_with_ensemble_entity(
        src_dir, tmp_dir, cfg.ensemble_entity, dry_run=cfg.dry_run
    )
    if not cfg.dry_run:
        update_dataset(
            cfg.kaggle_dataset.user_name,
            cfg.kaggle_dataset.title,
            tmp_dir,
            cfg.create,
        )

    if not cfg.dry_run:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
