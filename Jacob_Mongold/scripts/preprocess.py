"""Preprocess patient MRIs: resample -> normalize -> crop/pad -> save as .pt tensors."""

import argparse
import json
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from awhgcn.data.preprocessing import get_base_transforms

# keywords that show up in SeriesDescription for T1-weighted scans
_T1_KEYWORDS = {"t1", "spgr", "fspgr", "mprage", "bravo", "tfe", "irspgr", "ir-spgr"}
# anything in here is not a structural T1 — skip it
_EXCLUDE_KEYWORDS = {"t2", "flair", "dti", "dwi", "adc", "bold", "swi", "pd", "stir"}


def _score_nifti(nifti_path):
    """Score a NIfTI file so we can pick the best T1 when there are multiple candidates."""
    json_path = nifti_path.with_suffix(".json")
    if not json_path.exists():
        return (False, 0, False)
    try:
        meta = json.loads(json_path.read_text())
    except Exception:
        return (False, 0, False)

    desc = meta.get("SeriesDescription", "").lower()
    is_3d = meta.get("MRAcquisitionType", "") == "3D"
    thickness = float(meta.get("SliceThickness", 99))

    has_t1 = any(k in desc for k in _T1_KEYWORDS)
    has_exclude = any(k in desc for k in _EXCLUDE_KEYWORDS)
    if has_exclude:
        return (False, -thickness, False)
    # sort priority: 3D > thin slices > T1 keyword in name
    return (is_3d, -thickness, has_t1)


def find_t1(patient_dir):
    """Find the best T1-weighted NIfTI in a patient's folder using JSON sidecar metadata."""
    candidates = list(patient_dir.rglob("*.nii")) + list(patient_dir.rglob("*.nii.gz"))
    if not candidates:
        return None
    candidates.sort(key=_score_nifti, reverse=True)
    best = candidates[0]
    # if the best candidate still scored all-False (no metadata), just use it anyway
    if _score_nifti(best) == (False, 0, False) and len(candidates) > 1:
        best = candidates[0]
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n-patients", type=int, default=None, help="Cap for dry-run")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    mri_dir = Path(cfg.cohort.mri_dir)
    out_dir = Path(cfg.paths.preprocessed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mri_dir.exists():
        print(f"MRI directory not found: {mri_dir}")
        print("Run scripts/download_sample.py first.")
        return

    patient_dirs = [d for d in mri_dir.iterdir() if d.is_dir()]
    if args.n_patients:
        patient_dirs = patient_dirs[: args.n_patients]
    print(f"Processing {len(patient_dirs)} patient directories -> {out_dir}/")

    tf = get_base_transforms()
    success = skipped = failed = 0

    for pdir in tqdm(patient_dirs, desc="Preprocessing"):
        naccid = pdir.name
        out_path = out_dir / f"{naccid}.pt"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        nifti = find_t1(pdir)
        if nifti is None:
            print(f"  No NIfTI found: {naccid}")
            failed += 1
            continue

        try:
            data = tf({"image": str(nifti)})
            vol = data["image"]
            assert vol.shape == (1, 96, 96, 96), f"Shape mismatch: {vol.shape}"
            torch.save(vol.as_tensor().contiguous(), out_path)
            success += 1
        except Exception as e:
            print(f"  Failed {naccid}: {e}")
            failed += 1

    print(f"\nDone: {success} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
