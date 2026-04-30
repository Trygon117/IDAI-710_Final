#!/usr/bin/env python3
"""
List actual S3 keys in the NIfTI directory, extract NACCIDs from filenames,
cross-reference with our cohort, and download the ones that match.

Usage:
  python scripts/s3_find_mri.py --list-only     # scan S3 and show match stats
  python scripts/s3_find_mri.py --download       # scan then download matches
"""

import argparse
import re
import zipfile
from pathlib import Path
from collections import defaultdict
import boto3
import pandas as pd
from tqdm import tqdm

BUCKET   = "naccmri-quickaccess-sub"
# check both the within-1yr folder (preferred) and the full archive
PREFIXES = [
    "investigator/MRI/within1yr/nifti/",
    "investigator/MRI/all/nifti/",
]
MRI_CSV      = "data/raw/tabular/investigator_mri_nacc72.csv"
MANIFEST_CSV = "data/manifests/sampled_patients.csv"
NIFTI_DIR    = "data/mri_nifti_data"


# NACCID is always 6 digits preceded by "NACC" in the filename
_NACC_RE = re.compile(r"NACC(\d{6})", re.IGNORECASE)

def extract_naccid_from_key(key):
    """Pull the NACCID out of an S3 key like .../NACC123456_something.zip"""
    fname = key.split("/")[-1]
    m = _NACC_RE.search(fname)
    if m:
        return f"NACC{m.group(1)}"
    return None


def list_nifti_keys(s3_client, max_keys=None):
    """Page through all NIfTI keys in S3 and return a {naccid: [key, ...]} mapping."""
    mapping = defaultdict(list)
    all_keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for prefix in PREFIXES:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                naccid = extract_naccid_from_key(key)
                if naccid:
                    mapping[naccid].append(key)
                all_keys.append(key)
                if max_keys and len(all_keys) >= max_keys:
                    return mapping, all_keys
    return mapping, all_keys


def download_and_extract(s3_client, naccid, s3_key, out_dir):
    patient_dir = Path(out_dir) / naccid
    if patient_dir.exists() and any(patient_dir.rglob("*.nii*")):
        return "skipped"
    zip_path = Path(out_dir) / f"{naccid}_tmp.zip"
    try:
        s3_client.download_file(BUCKET, s3_key, str(zip_path))
        patient_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(patient_dir)
        return "ok"
    except Exception as e:
        return f"error: {e}"
    finally:
        zip_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list-only", action="store_true",
                   help="Scan S3 and report match stats without downloading")
    p.add_argument("--download", action="store_true",
                   help="Scan S3 then download matching files")
    p.add_argument("--save-map", default="data/manifests/s3_naccid_map.csv",
                   help="Save the NACCID->S3 key mapping to CSV")
    p.add_argument("--max-keys", type=int, default=None,
                   help="Cap total keys scanned (useful for quick tests)")
    args = p.parse_args()

    if not args.list_only and not args.download:
        p.print_help()
        return

    s3 = boto3.client("s3")

    # load our cohort so we can check how many of our patients are actually on S3
    if Path(MANIFEST_CSV).exists():
        sampled_ids = set(pd.read_csv(MANIFEST_CSV)["NACCID"].unique())
    else:
        sampled_ids = set(pd.read_csv(MRI_CSV)["NACCID"].unique())
    print(f"Cohort size: {len(sampled_ids)} patients")

    print(f"Scanning S3 bucket {BUCKET} for NIfTI keys...")
    key_map, all_keys = list_nifti_keys(s3, max_keys=args.max_keys)
    print(f"  Found {len(all_keys)} total keys, {len(key_map)} with parseable NACCID")

    matched = {nid: key_map[nid] for nid in sampled_ids if nid in key_map}
    missing = sampled_ids - set(key_map.keys())
    print(f"  Matched: {len(matched)}/{len(sampled_ids)} cohort patients have keys on S3")
    print(f"  Missing: {len(missing)} patients not found in S3")

    # save the full S3 map (all patients, not just our cohort) — used by resample_from_s3.py
    all_rows = []
    for nid, keys in key_map.items():
        for k in keys:
            all_rows.append({"NACCID": nid, "s3_key": k})
    all_map_path = Path(args.save_map).parent / "s3_all_naccid_map.csv"
    all_map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(all_map_path, index=False)
    print(f"  Full S3 map ({len(key_map)} NACCIDs) saved -> {all_map_path}")

    # also save the cohort-intersection subset for convenience
    if matched:
        rows = []
        for nid, keys in matched.items():
            for k in keys:
                rows.append({"NACCID": nid, "s3_key": k})
        map_path = Path(args.save_map)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(map_path, index=False)
        print(f"  Cohort-matched map ({len(matched)} NACCIDs) saved -> {map_path}")

    if args.list_only or not matched:
        if not matched:
            print("\nNo matches found. Check bucket/prefix or NACCID format in filenames.")
            print("\nSample S3 keys (first 20):")
            for k in all_keys[:20]:
                print(f"  {k}")
        return

    # download mode
    print(f"\nDownloading {len(matched)} patients...")
    Path(NIFTI_DIR).mkdir(parents=True, exist_ok=True)
    success = failed = skipped = 0

    for nid, keys in tqdm(matched.items(), desc="Downloading"):
        # use the first key for each patient (within1yr comes first since it's listed first)
        result = download_and_extract(s3, nid, keys[0], NIFTI_DIR)
        if result == "ok":
            success += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"  FAILED {nid}: {result}")

    print(f"\nDone: {success} downloaded, {skipped} skipped, {failed} failed")
    print(f"MRI data -> {NIFTI_DIR}/")


if __name__ == "__main__":
    main()
