import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.config import MainConfig
from src.constant import EEG_PROBES, PROBES
from src.preprocess import load_eeg, process_cqf, process_eeg
from src.proc_util import trace


def save_eeg(eeg_id: str, eeg_df: pl.DataFrame, output_dir: Path):
    eeg = eeg_df.select(PROBES).to_numpy()

    output_file_path = output_dir / str(eeg_id)
    if not output_file_path.exists():
        output_file_path.mkdir(parents=True)

    info = np.finfo(np.float16)
    eeg = eeg.clip(info.min, info.max).astype(np.float16)
    np.save(output_file_path / "eeg.npy", eeg)


def save_cqf(eeg_id: str, eeg_df: pl.DataFrame, output_dir: Path):
    cqf = (
        eeg_df.select(f"CQF-{probe}" for probe in EEG_PROBES)
        .to_numpy()
        .astype(np.float16)
    )
    mask = (
        eeg_df.select(f"mask-{probe}" for probe in EEG_PROBES)
        .to_numpy()
        .astype(np.uint8)
    )

    output_file_path = output_dir / str(eeg_id)
    if not output_file_path.exists():
        output_file_path.mkdir(parents=True)

    np.save(output_file_path / "cqf.npy", cqf)
    np.save(output_file_path / "mask.npy", mask)


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")
    output_dir = Path(cfg.env.output_dir)

    # ディレクトリが存在する場合は削除
    if (cfg.preprocess.cleanup) and (output_dir.exists()):
        shutil.rmtree(output_dir)
        print(f"Removed {cfg.phase} dir: {output_dir}")

    eeg_ids = metadata["eeg_id"].unique().to_numpy()
    if cfg.debug:
        num_samples = 100
        eeg_ids = eeg_ids[: min(len(eeg_ids), num_samples)]

    tag = " (with cqf)" if cfg.preprocess.process_cqf else ""

    with trace(f"process eeg{tag}"):
        for eeg_id in tqdm(eeg_ids, total=eeg_ids.shape[0]):
            eeg_df = load_eeg(eeg_id, data_dir=data_dir)
            eeg = process_eeg(eeg_df)
            eeg_df = pl.DataFrame(
                {probe: pl.Series(v) for probe, v in zip(PROBES, eeg.T)}
            )
            save_eeg(eeg_id, eeg_df, Path(cfg.env.output_dir))

            if cfg.preprocess.process_cqf:
                eeg_df = process_cqf(eeg_df)
                save_cqf(eeg_id, eeg_df, Path(cfg.env.output_dir))


if __name__ == "__main__":
    main()
