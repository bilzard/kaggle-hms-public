import json
import shutil
from pathlib import Path
from typing import Any

import click
from kaggle.api.kaggle_api_extended import KaggleApi


def copy_files_with_globs(
    source_dir: Path, dest_dir: Path, glob_patterns: list[str], dry_run: bool
):
    """
    automatically copy files with the specified glob patterns in the source directory to the destination directory
    """
    for glob_pattern in glob_patterns:
        for source_path in source_dir.rglob(glob_pattern):
            if "debug" in source_path.parts:
                continue
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


@click.command()
@click.option("--title", "-t", default="HMS-model")
@click.option("--src_dir", "-s", type=Path, default=Path("./data/train"))
@click.option("--glob_patterns", "-g", multiple=True, default=["last_model.pth"])
@click.option("--user_name", "-u", default="tatamikenn")
@click.option("--create", "-c", is_flag=True)
@click.option("--dry_run", "-d", is_flag=True)
def main(
    title: str,
    src_dir: Path,
    glob_patterns: list[str],
    user_name: str,
    create: bool = False,
    dry_run: bool = False,
):
    """automatically create and upload dataset on kaggle based on the glob patterns in the source directory

    Args:
        title (str): title of the dataset
        src_dir (Path): source directory of the dataset
        glob_pattern (list[str], optional): glob pattern of the files to be uploaded
        user_name (str, optional): kaggle user name
        create (bool, optional): create new dataset or not
    """
    assert src_dir.exists()
    assert len(glob_patterns) > 0

    if dry_run:
        print("=== dry run mode ===")
    tmp_dir = Path("./tmp")
    if not dry_run:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    copy_files_with_globs(src_dir, tmp_dir, glob_patterns, dry_run)
    if not dry_run:
        update_dataset(user_name, title, tmp_dir, create)

    if not dry_run:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
