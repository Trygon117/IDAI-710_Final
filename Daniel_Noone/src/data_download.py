# Imports

import subprocess
import os
import re
import zipfile
import json
from tqdm import tqdm
import shutil

#import preprocessing as preproc


import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# Loading data

local_path = "mri_nifti_data/"
os.makedirs(local_path, exist_ok=True)

# Check how many files we currently have
existing_files = [f for f in os.listdir(local_path)]

print(f"Currently have {len(existing_files)} files in the local directory.")

# If we already have our 200 files, we can skip the entire download process
if len(existing_files) >= 200:
    print(f"All {len(existing_files)} files are already downloaded. Skipping AWS execution.")
else:
    print("Starting master download process...")

    # 1. Inject credentials for the Mixed Protocol dataset
    my_env = os.environ.copy()
    my_env["AWS_ACCESS_KEY_ID"] = "REDACTED"
    my_env["AWS_SECRET_ACCESS_KEY"] = "REDACTED"

    # 2. Fetch the absolute ground truth list of NIFTI files from the server
    print("Fetching master list of available NIFTI files from AWS...")
    s3_base_path = "s3://naccmri-quickaccess-sub/investigator/MRI/within1yr/nifti/"
    command_ls = ["aws", "s3", "ls", s3_base_path]
    result_ls = subprocess.run(command_ls, env=my_env, capture_output=True, text=True)

    if result_ls.returncode != 0:
        print("Error fetching list from AWS:", result_ls.stderr)
    else:
        # Extract all the NACC IDs from the server output using a regular expression
        available_ids = set(re.findall(r'NACC\d+', result_ls.stdout))

        # Load the tabular clinical data
        clinical_df = pd.read_csv('data/investigator_nacc72.csv', low_memory=False)
        mri_df = pd.read_csv('data/investigator_mri_nacc72.csv', low_memory=False)

        # Find common patient IDs
        common_ids = set(clinical_df['NACCID']).intersection(set(mri_df['NACCID']))
        print(f"Total overlapping patients {len(common_ids)}")

        guaranteed_df = clinical_df[clinical_df['NACCID'].isin(common_ids) & clinical_df['NACCID'].isin(available_ids)]

        # Drop duplicate visits, keeping only the most recent visit to get the most definitive diagnosis
        guaranteed_df = guaranteed_df.sort_values('VISITYR', ascending=False).drop_duplicates(subset=['NACCID'])

        # Define our strict exclusion masks to prevent comorbidities
        # We want to ensure a patient only has a 1 in their specific disease column
        disease_cols = ['NACCALZD', 'NACCVASC', 'NACCLBDE', 'NACCFTD']
        has_any_disease = guaranteed_df[disease_cols].sum(axis=1)

        # 1. Cognitively Normal (CN) - Label 0
        # Must be normal and strictly NOT diagnosed with any of the four diseases
        cn_mask = (guaranteed_df['NORMCOG'] == 1) & \
                  (guaranteed_df['NACCALZD'] != 1) & \
                  (guaranteed_df['NACCVASC'] != 1) & \
                  (guaranteed_df['NACCLBDE'] != 1) & \
                  (guaranteed_df['NACCFTD'] != 1)

        cn_df = guaranteed_df[cn_mask].copy()
        cn_df['DIAGNOSIS_LABEL'] = 0

        # 2. Mild Cognitive Impairment (MCI) - Label 1
        # Updated to 3 based on NACC UDS data dictionary
        mci_mask = (guaranteed_df['NACCUDSD'] == 3) & \
                   (guaranteed_df['NACCALZD'] != 1) & \
                   (guaranteed_df['NACCVASC'] != 1) & \
                   (guaranteed_df['NACCLBDE'] != 1) & \
                   (guaranteed_df['NACCFTD'] != 1)

        mci_df = guaranteed_df[mci_mask].copy()
        mci_df['DIAGNOSIS_LABEL'] = 1

        # 3. Alzheimer disease (AD) - Label 2
        # Must explicitly have AD and strictly NOT have the other three
        ad_mask = (guaranteed_df['NACCALZD'] == 1) & \
                  (guaranteed_df['NACCVASC'] != 1) & \
                  (guaranteed_df['NACCLBDE'] != 1) & \
                  (guaranteed_df['NACCFTD'] != 1)

        ad_df = guaranteed_df[ad_mask].copy()
        ad_df['DIAGNOSIS_LABEL'] = 2

        # 4. Vascular Dementia (VaD) - Label 3
        vad_mask = (guaranteed_df['NACCVASC'] == 1) & \
                   (guaranteed_df['NACCALZD'] != 1) & \
                   (guaranteed_df['NACCLBDE'] != 1) & \
                   (guaranteed_df['NACCFTD'] != 1)

        vad_df = guaranteed_df[vad_mask].copy()
        vad_df['DIAGNOSIS_LABEL'] = 3

        # 5. Lewy Body Dementia (LBD) - Label 4
        lbd_mask = (guaranteed_df['NACCLBDE'] == 1) & \
                   (guaranteed_df['NACCALZD'] != 1) & \
                   (guaranteed_df['NACCVASC'] != 1) & \
                   (guaranteed_df['NACCFTD'] != 1)

        lbd_df = guaranteed_df[lbd_mask].copy()
        lbd_df['DIAGNOSIS_LABEL'] = 4

        # 6. Frontotemporal Dementia (FTD) - Label 5
        ftd_mask = (guaranteed_df['NACCFTD'] == 1) & \
                   (guaranteed_df['NACCALZD'] != 1) & \
                   (guaranteed_df['NACCVASC'] != 1) & \
                   (guaranteed_df['NACCLBDE'] != 1)

        ftd_df = guaranteed_df[ftd_mask].copy()
        ftd_df['DIAGNOSIS_LABEL'] = 5

        print(f"Available pure CN: {len(cn_df)}")
        print(f"Available pure MCI: {len(mci_df)}")
        print(f"Available pure AD: {len(ad_df)}")
        print(f"Available pure VaD: {len(vad_df)}")
        print(f"Available pure LBD: {len(lbd_df)}")
        print(f"Available pure FTD: {len(ftd_df)}")

        # Standardize the cohort size based on the smallest available class to ensure perfect balance
        min_class_size = min(len(cn_df), len(mci_df), len(ad_df), len(vad_df), len(lbd_df), len(ftd_df))

        final_cohort_df = pd.concat([
            cn_df.sample(n=min_class_size, random_state=42),
            mci_df.sample(n=min_class_size, random_state=42),
            ad_df.sample(n=min_class_size, random_state=42),
            vad_df.sample(n=min_class_size, random_state=42),
            lbd_df.sample(n=min_class_size, random_state=42),
            ftd_df.sample(n=min_class_size, random_state=42)
        ])

        final_ids = final_cohort_df['NACCID'].tolist()

        # Check your extracted folders against your new master list
        for folder_name in os.listdir(local_path):
            folder_path = os.path.join(local_path, folder_name)

            # If the folder is not in your new finalized cohort, delete it
            if os.path.isdir(folder_path) and folder_name not in final_ids:
                print(f"Removing unused patient data {folder_name}")
                shutil.rmtree(folder_path)

        # 5. Download the files
        print("Downloading files...")

        # Wrap the loop with tqdm to create the progress bar
        progress_bar = tqdm(final_ids, desc="Downloading NIFTI files")

        for pid in progress_bar:
            # Only download the file if it does not already exist in the folder
            if not any(pid in f for f in existing_files):
                # Update the extra information space next to the loading bar
                progress_bar.set_postfix(current_patient=pid)

                command_cp = [
                    "aws", "s3", "cp", s3_base_path, local_path,
                    "--recursive", "--exclude", "*", "--include", f"*{pid}*"
                ]
                subprocess.run(command_cp, env=my_env, capture_output=True)

        print("Master download complete. You have exactly 200 guaranteed files.")