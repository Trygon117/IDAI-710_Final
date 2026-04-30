#!/usr/bin/env python3
"""Download NIfTI MRI zips from S3 for a sampled set of patients."""

import argparse
import zipfile
from pathlib import Path
import boto3
import botocore
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BUCKET = "naccmri-quickaccess-sub"
MRI_CSV = "data/raw/tabular/investigator_mri_nacc72.csv"
MANIFEST_OUT = "data/manifests/sampled_patients.csv"
NIFTI_DIR = "data/mri_nifti_data"


def _download_and_extract(s3_client, naccid, s3_key, out_dir):
    patient_dir = Path(out_dir) / naccid
    # skip if we already have the NIfTI files for this patient
    if patient_dir.exists() and any(patient_dir.rglob("*.nii*")):
        return True

    zip_path = Path(out_dir) / f"{naccid}_tmp.zip"
    try:
        s3_client.download_file(BUCKET, s3_key, str(zip_path))
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return False
        raise
    try:
        patient_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(patient_dir)
        return True
    except Exception as e:
        print(f"  Extract error {naccid}: {e}")
        return False
    finally:
        zip_path.unlink(missing_ok=True)  # always clean up the temp zip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-patients", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Save manifest but skip downloads")
    args = p.parse_args()

    mri_df = pd.read_csv(MRI_CSV, low_memory=False).drop_duplicates("NACCID")
    print(f"Loaded {len(mri_df)} unique patients with MRI filenames from {MRI_CSV}")

    sampled = mri_df.sample(n=min(args.n_patients, len(mri_df)), random_state=args.seed)
    print(f"Sampled {len(sampled)} patients (seed={args.seed})")

    Path(MANIFEST_OUT).parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(MANIFEST_OUT, index=False)
    print(f"Sample manifest saved to: {MANIFEST_OUT}")

    if args.dry_run:
        return

    nifti_dir = Path(NIFTI_DIR)
    nifti_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    # build task list — skip patients with no MRI filename in the CSV
    tasks = [
        (row["NACCID"], f"investigator/MRI/within1yr/nifti/{row['NACCMRFI']}")
        for _, row in sampled.iterrows()
        if pd.notna(row.get("NACCMRFI"))
    ]

    success, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_download_and_extract, s3, nid, key, nifti_dir): nid
                for nid, key in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            nid = futs[fut]
            try:
                ok = fut.result()
                (success if ok else failed).append(nid)
                if not ok:
                    print(f"  WARNING: Failed for {nid}")
            except Exception as e:
                print(f"  ERROR {nid}: {e}")
                failed.append(nid)

    print(f"\nDownload complete: {len(success)} success, {len(failed)} failed")
    # save a manifest of just the patients we successfully downloaded
    success_df = sampled[sampled["NACCID"].isin(success)]
    success_df.to_csv(Path(MANIFEST_OUT).parent / "downloaded_patients.csv", index=False)
    print(f"Downloaded manifest -> data/manifests/downloaded_patients.csv")


if __name__ == "__main__":
    main()
