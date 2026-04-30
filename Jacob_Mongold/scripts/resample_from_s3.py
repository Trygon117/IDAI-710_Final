#!/usr/bin/env python3
"""
Cross-reference all S3-confirmed NACCIDs with the tabular CSV,
assign 3-group labels, report availability, resample cohort, and download.

Workflow:
  1. Load data/manifests/s3_all_naccid_map.csv  (built by s3_find_mri.py --list-only)
  2. Run load_cohort() filtered to only confirmed S3 patients
  3. Print per-group counts and warn if < 100 in any group
  4. Save new manifest -> data/manifests/sampled_patients.csv (overwrites)
  5. Download unless --dry-run

Usage:
  # Step 1: build the full S3 map (only needed once)
  python scripts/s3_find_mri.py --list-only

  # Step 2: resample + download
  python scripts/resample_from_s3.py
  python scripts/resample_from_s3.py --dry-run   # report only, no downloads
"""

import argparse
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import botocore
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from awhgcn.data.cohort import load_cohort

BUCKET       = "naccmri-quickaccess-sub"
TABULAR_CSV  = "data/raw/tabular/investigator_nacc72.csv"
ALL_MAP_CSV  = "data/manifests/s3_all_naccid_map.csv"
MANIFEST_OUT = "data/manifests/sampled_patients.csv"
NIFTI_DIR    = "data/mri_nifti_data"


def _download_and_extract(s3_client, naccid, s3_key, out_dir):
    patient_dir = Path(out_dir) / naccid
    # already have it — don't re-download
    if patient_dir.exists() and any(patient_dir.rglob("*.nii*")):
        return "skipped"
    zip_path = Path(out_dir) / f"{naccid}_tmp.zip"
    try:
        s3_client.download_file(BUCKET, s3_key, str(zip_path))
        patient_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(patient_dir)
        return "ok"
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        return f"s3-error-{code}"
    except Exception as e:
        return f"error: {e}"
    finally:
        zip_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-group", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Report availability and save manifest but skip downloads")
    p.add_argument("--all-map", default=ALL_MAP_CSV,
                   help="Path to s3_all_naccid_map.csv (from s3_find_mri.py --list-only)")
    args = p.parse_args()

    # need the S3 map first — run s3_find_mri.py --list-only if this is missing
    if not Path(args.all_map).exists():
        print(f"ERROR: {args.all_map} not found.")
        print("Run: python scripts/s3_find_mri.py --list-only")
        sys.exit(1)

    s3_map = pd.read_csv(args.all_map)
    confirmed_naccids = set(s3_map["NACCID"].unique())
    print(f"S3-confirmed NACCIDs: {len(confirmed_naccids)}")

    # prefer the within1yr scan if available — more likely to match the tabular visit date
    def best_key(keys):
        for k in keys:
            if "within1yr" in k:
                return k
        return keys[0]

    key_by_naccid = (
        s3_map.groupby("NACCID")["s3_key"]
        .apply(lambda ks: best_key(ks.tolist()))
        .to_dict()
    )

    # trick: pass confirmed IDs as a fake mri_csv so load_cohort only samples from patients on S3
    tmp_mri_csv = Path("data/manifests/_s3_confirmed_ids.csv")
    pd.DataFrame({"NACCID": list(confirmed_naccids)}).to_csv(tmp_mri_csv, index=False)

    print(f"Loading cohort from {TABULAR_CSV} filtered to S3-confirmed patients...")
    cohort_df, used_cols = load_cohort(
        TABULAR_CSV,
        mri_csv=str(tmp_mri_csv),
        n_per_group=args.n_per_group,
        seed=args.seed,
    )
    tmp_mri_csv.unlink(missing_ok=True)

    from awhgcn.data.cohort import LABEL_MAP
    for g in LABEL_MAP:
        n = (cohort_df["group"] == g).sum()
        flag = " <<< WARNING: < 100" if n < args.n_per_group else ""
        print(f"  Group {g}: {n} patients{flag}")

    cohort_df["s3_key"] = cohort_df["NACCID"].map(key_by_naccid)

    Path(MANIFEST_OUT).parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(MANIFEST_OUT, index=False)
    print(f"Cohort manifest saved -> {MANIFEST_OUT}")

    if args.dry_run:
        print("--dry-run: skipping downloads.")
        return

    tasks = [
        (row["NACCID"], row["s3_key"])
        for _, row in cohort_df.iterrows()
        if pd.notna(row.get("s3_key"))
    ]
    print(f"\nDownloading {len(tasks)} patients ({args.workers} workers)...")
    Path(NIFTI_DIR).mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    results = {"ok": [], "skipped": [], "failed": []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_download_and_extract, s3, nid, key, NIFTI_DIR): nid
            for nid, key in tasks
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            nid = futs[fut]
            try:
                status = fut.result()
                if status in ("ok", "skipped"):
                    results[status].append(nid)
                else:
                    results["failed"].append(nid)
                    print(f"  FAILED {nid}: {status}")
            except Exception as e:
                results["failed"].append(nid)
                print(f"  ERROR {nid}: {e}")

    ok = len(results["ok"])
    sk = len(results["skipped"])
    fa = len(results["failed"])
    print(f"\nDone: {ok} downloaded, {sk} already present, {fa} failed")
    print(f"MRI data -> {NIFTI_DIR}/")

    # save a clean manifest of just the patients we have data for
    from awhgcn.data.cohort import LABEL_MAP
    downloaded_ids = set(results["ok"]) | set(results["skipped"])
    dl_df = cohort_df[cohort_df["NACCID"].isin(downloaded_ids)]
    dl_path = Path(MANIFEST_OUT).parent / "downloaded_patients.csv"
    dl_df.to_csv(dl_path, index=False)
    print(f"Downloaded manifest -> {dl_path}")
    for g in LABEL_MAP:
        n = (dl_df["group"] == g).sum()
        print(f"  Group {g}: {n} downloaded")


if __name__ == "__main__":
    main()
