import json
import os
import re
import torch
from torch.utils.data import Dataset
import monai.transforms as mt
import nibabel as nib

def get_mri_transforms():
    # Compose chains multiple operations together in order
    transform_pipeline = mt.Compose([
        # Loads the NIFTI file and extracts the 3D image data
        mt.LoadImaged(keys=["image"]),
        
        # PyTorch requires the color or data channel to be the first dimension
        mt.EnsureChannelFirstd(keys=["image"]),
        
        # 1. Force every scan into the exact same anatomical orientation
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        
        # 2. Normalize the physical size of the voxels so 1 pixel = 1x1x1 mm
        mt.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        
        # Resizes all 3D volumes to a uniform 96x96x96 box
        # We use a smaller size here to ensure it trains quickly
        mt.Resized(keys=["image"], spatial_size=(96, 96, 96)),
        
        # Scales the pixel intensity values to be between 0 and 1
        mt.ScaleIntensityd(keys=["image"])
    ])
    
    return transform_pipeline

def find_nifti_files(patient_id, base_path="data/mri_nifti_data/"):
    patient_dir = os.path.join(base_path, patient_id)
    
    # The walk function digs through every folder and subfolder inside the directory
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            # We only care about the actual neuroimaging files
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                print(file)

def inspect_nifti_headers(patient_id, base_path="data/mri_nifti_data/"):
    patient_dir = os.path.join(base_path, patient_id)
    
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                
                # Load the image using nibabel to access the metadata
                img = nib.load(file_path)
                header = img.header
                
                # Extract and print the description field from the header
                print(f"File {file}")
                print(f"Description {header['descrip']}")
                print("")

def select_t1_scan(patient_id, base_path="data/mri_nifti_data/"):
    patient_dir = os.path.join(base_path, patient_id)
    
    t1_file_path = None
    lowest_te = float('inf')
    
    # Walk the directory looking for JSON sidecar files instead of NIFTI files
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                
                # Open and parse the JSON file
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Safely extract the EchoTime, defaulting to infinity if the key is missing
                te_value = metadata.get("EchoTime", float('inf'))
                
                # Compare and store the lowest TE value
                if te_value < lowest_te:
                    lowest_te = te_value
                    
                    # Construct the path to the matching image file
                    # We strip the '.json' extension (the last 5 characters) from the name
                    base_name = file[:-5] 
                    
                    # NIFTI files might be compressed (.nii.gz) or uncompressed (.nii)
                    # We check the hard drive to see which one actually exists
                    if os.path.exists(os.path.join(root, base_name + '.nii.gz')):
                        t1_file_path = os.path.join(root, base_name + '.nii.gz')
                    else:
                        t1_file_path = os.path.join(root, base_name + '.nii')
                        
    return t1_file_path

def extract_metadata_tensor(json_path, max_mag_strength=15000.0, max_slice_thickness=16.0, max_echo_time=0.564):
    with open(json_path, 'r') as f:
        meta = json.load(f)
        
    # Normalize using the true dataset maximums we just calculated
    mag_strength = meta.get("MagneticFieldStrength", 15000.0) / max_mag_strength
    slice_thick = meta.get("SliceThickness", 1.5) / max_slice_thickness
    echo_time = meta.get("EchoTime", 0.0) / max_echo_time
    
    meta_tensor = torch.tensor([mag_strength, slice_thick, echo_time], dtype=torch.float32)
    
    return meta_tensor

def extract_and_save_nodes(dataloader, encoder, device, filename):
    all_nodes = []
    all_clin = []
    all_labels = []

    # Disable gradient calculation to save memory
    with torch.no_grad():
        for images, meta_tokens, clin_tensors, batch_labels in dataloader:
            images = images.to(device)
            
            # Pass the 3D brain through the encoder to get our single node vector
            nodes = encoder(images)
            
            # Move the data back to the CPU and store it
            all_nodes.append(nodes.cpu())
            all_clin.append(clin_tensors)
            all_labels.append(batch_labels)

    # Glue the batches together into single large tensors
    final_nodes = torch.cat(all_nodes, dim=0)
    final_clin = torch.cat(all_clin, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # Save the tensors to the hard drive
    torch.save({
        'nodes': final_nodes,
        'clinical': final_clin,
        'labels': final_labels
    }, filename)
    
    print(f"Saved {len(final_nodes)} patients to {filename}")

class HyperFuseDataset(Dataset):
    def __init__(self, patient_ids, labels, clinical_features, max_vals, base_path="data/mri_nifti_data/"):
        self.patient_ids = patient_ids
        self.labels = labels
        self.clinical_features = clinical_features 
        self.max_vals = max_vals
        self.base_path = base_path
        
        self.transform = get_mri_transforms()

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        
        # Grab the correct T1 scan
        image_path = select_t1_scan(pid, self.base_path)
        
        # Reconstruct the JSON path by swapping the file extension
        if image_path.endswith('.nii.gz'):
            json_path = image_path.replace('.nii.gz', '.json')
        else:
            json_path = image_path.replace('.nii', '.json')
            
        # Run the image through the MONAI assembly line
        image_dict = {"image": image_path}
        transformed_image = self.transform(image_dict)["image"]
        
        # Grab the normalized metadata token using the dataset's stored maximums
        meta_token = extract_metadata_tensor(
            json_path,
            max_mag_strength=self.max_vals['mag'],
            max_slice_thickness=self.max_vals['slice'],
            max_echo_time=self.max_vals['te']
        )

        # Grab the pre-processed clinical features for this specific patient
        clinical_tensor = self.clinical_features[idx]
        
        # Convert the clinical diagnosis label to a standard PyTorch tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Return all four components
        return transformed_image, meta_token, clinical_tensor, label_tensor