import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from src.config import MainConfig
from src.data_util import preload_cqf, preload_eegs
from src.dataset.eeg import PerEegDataset, get_valid_loader
from src.model.feature_extractor.wave2spec import Wave2Spectrogram
from src.preprocess import process_label
from src.proc_util import trace


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")
    metadata = process_label(metadata)
    output_dir = Path(cfg.env.output_dir)

    if (not cfg.dry_run) and (cfg.cleanup) and (output_dir.exists()):
        shutil.rmtree(output_dir)
        print(f"Removed {cfg.phase} dir: {output_dir}")

    preprocess_dir = working_dir / "preprocess" / cfg.phase
    eeg_ids = sorted(metadata["eeg_id"].unique().to_list())

    with trace("load eeg"):
        id2eeg = preload_eegs(eeg_ids, preprocess_dir)
        id2cqf = preload_cqf(eeg_ids, preprocess_dir)

    with trace("generate spectrograms"):
        if cfg.debug:
            metadata = metadata.head(10)

        dataset = PerEegDataset(metadata, id2eeg, id2cqf)
        dataloader = get_valid_loader(
            dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
        )
        model = Wave2Spectrogram()
        device = torch.device("cuda")
        model.to(device)

        for batch in tqdm(dataloader, total=len(dataloader)):
            eeg, eeg_id, cqf = (
                batch["eeg"],
                batch["eeg_id"],
                batch["cqf"],
            )
            eeg = eeg.to(device)
            output = model(eeg, mask=cqf, apply_mask=False)

            eeg_id = eeg_id[0].detach().cpu().numpy()
            signal = output["signal"][0].detach().cpu().numpy().astype(np.float16)
            channel_mask = (
                output["channel_mask"][0].detach().cpu().numpy().astype(np.float16)
            )
            spectrogram = (
                output["spectrogram"][0].detach().cpu().numpy().astype(np.float16)
            )
            spec_mask = output["spec_mask"][0].detach().cpu().numpy().astype(np.float16)
            probe_pairs = output["probe_pairs"]
            probe_groups = output["probe_groups"]

            if not cfg.dry_run:
                eeg_dir = output_dir / "spectrogram" / str(eeg_id)
                if not eeg_dir.exists():
                    eeg_dir.mkdir(parents=True)

                np.save(eeg_dir / "spectrogram.npy", spectrogram)
                np.save(eeg_dir / "spec_mask.npy", spec_mask)
                np.save(eeg_dir / "channel_mask.npy", channel_mask)
                np.save(eeg_dir / "signal.npy", signal)

        if not cfg.dry_run:
            np.save(output_dir / "probe_pairs.npy", probe_pairs)
            np.save(output_dir / "probe_groups.npy", probe_groups)


if __name__ == "__main__":
    main()
