#!/usr/bin/env python3
"""Report cohort group sizes and MRI download status, then save the manifest."""

import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from awhgcn.data.cohort import load_cohort


def main():
    cfg = OmegaConf.load("configs/default.yaml")

    df, used_cols = load_cohort(
        cfg.cohort.tabular_csv, mri_csv=cfg.cohort.mri_csv,
        n_per_group=cfg.cohort.n_per_group, seed=cfg.seed,
    )

    # check which patients already have their scans preprocessed
    preprocessed_dir = Path(cfg.paths.preprocessed_dir)
    df["has_preprocessed"] = df["NACCID"].apply(
        lambda x: (preprocessed_dir / f"{x}.pt").exists()
    )

    from awhgcn.data.cohort import LABEL_MAP
    for g in LABEL_MAP:
        sub = df[df["group"] == g]
        has = sub["has_preprocessed"].sum()
        print(f"Group {g}: {len(sub):3d} patients | {has:3d} with preprocessed MRI")

    manifest_path = Path(cfg.cohort.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(manifest_path, index=False)
    print(f"\nManifest -> {manifest_path}")
    print(f"Tabular features: {used_cols}")


if __name__ == "__main__":
    main()
