import json
import os
import torch
from torch.utils.data import Dataset
import monai.transforms as mt


def get_mri_transforms():
    # Compose chains multiple operations together in order
    transform_pipeline = mt.Compose([
        # Loads the NIFTI file and extracts the 3D image data
        mt.LoadImaged(keys=["image"]),

        # PyTorch requires the color or data channel to be the first dimension
        mt.EnsureChannelFirstd(keys=["image"]),

        # forcing same Px orientation
        mt.Orientationd(keys=['image'], axcodes = "RAS"),

        # std vox spacing
        mt.Spacingd(keys=['image'], pixdim=(1.0,1.0,1.0), mode='bilinear'),

        # Resizes all 3D volumes to a uniform 96x96x96 box
        # We use a smaller size here to ensure it trains quickly for your one week deadline
        mt.Resized(keys=["image"], spatial_size=(96, 96, 96)),

        # Scales the pixel intensity values to be between 0 and 1
        mt.ScaleIntensityd(keys=["image"]),
        
        # guarantee pytorch tensor 
        mt.EnsureTyped(keys=['image'])
    ])

    return transform_pipeline




def select_t1_scan(patient_id, base_path="/naac_data/"):
    patient_dir = os.path.join(base_path, patient_id)

    t1_file_path = None
    lowest_te = float('inf')

    # walking dir looking for json files not nifti files
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)

                # open & parse json f
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                # extracting echotime, def to inf if key missing
                te_value = metadata.get("EchoTime", float('inf'))

                # choosing lowest te valk
                if te_value < lowest_te:
                    lowest_te = te_value

                    # const path to the matching image file
                    base_name = file[:-5] # stripping json extension from name

                    # nifti fs maybe compressed 
                    # checking to see which is there
                    if os.path.exists(os.path.join(root, base_name + '.nii.gz')):
                        t1_file_path = os.path.join(root, base_name + '.nii.gz')
                    else:
                        t1_file_path = os.path.join(root, base_name + '.nii')

    return t1_file_path


def assign_label(row):
    if row["NORMCOG"] == 1 and row["NACCALZD"] != 1 and row["NACCVASC"] != 1 and row["NACCLBDE"] != 1 and row["NACCFTD"] != 1:
        return 0  # CN

    if row["NACCUDSD"] == 3 and row["NACCALZD"] != 1 and row["NACCVASC"] != 1 and row["NACCLBDE"] != 1 and row["NACCFTD"] != 1:
        return 1  # MCI

    if row["NACCALZD"] == 1 and row["NACCVASC"] != 1 and row["NACCLBDE"] != 1 and row["NACCFTD"] != 1:
        return 2  # AD

    if row["NACCVASC"] == 1 and row["NACCALZD"] != 1 and row["NACCLBDE"] != 1 and row["NACCFTD"] != 1:
        return 3  # VaD

    if row["NACCLBDE"] == 1 and row["NACCALZD"] != 1 and row["NACCVASC"] != 1 and row["NACCFTD"] != 1:
        return 4  # LBD

    if row["NACCFTD"] == 1 and row["NACCALZD"] != 1 and row["NACCVASC"] != 1 and row["NACCLBDE"] != 1:
        return 5  # FTD

    return None  # ambiguous/comorbid, skip

    

class nacc_mri_ds(Dataset):
    def __init__(self, df, tab_cols, pid_mri):

        # store the df
        self.df = df.reset_index(drop = True)
        # store cols used as tab input
        self.tab_cols = tab_cols

        # store px id to mri path map3
        self.pid_to_mri = pid_mri

        # load mri preproc 
        self.transform = get_mri_transforms()

    def __len__(self): # getting num Pxs, gives torch dataset size
        return len(self.df)

    def __getitem__(self, idx): # from index, ret  sample of img, tabular, label
        r = self.df.iloc[idx] # one row (one Px)
        pid = r["NACCID"] # getting Px id number

        # getting mri file path for Px
        img_path = self.pid_to_mri[pid]

        # loading and preproc 3D mri scan
        img_dict = {"image": img_path} # found out that monai needs a dict of image and path - SPECIFICALLY 'IMAGE'
        img = self.transform(img_dict)["image"]

        # ensuring float bc how model defined
        img = img.float()

        # get tab fs for Px
        tab = torch.tensor(
            r[self.tab_cols].values.astype("float32"),
            dtype = torch.float32)

        # getting label for given Px
        l = torch.tensor(r["label"], dtype = torch.long)

        return img, tab, l


