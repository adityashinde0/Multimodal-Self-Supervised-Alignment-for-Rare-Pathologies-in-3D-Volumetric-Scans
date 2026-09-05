import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class VolumeReportDataset(Dataset):
    """
    Dataset loader for paired 3D volumetric scans (.bin raw float32)
    and unstructured clinical radiology reports.
    """
    def __init__(self, data_dir=".", report_file="radiology_reports.json", 
                 augment=False, target_shape=(16, 16, 16), indices=None):
        super().__init__()
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.augment = augment
        
        report_path = os.path.join(data_dir, report_file)
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file not found at: {report_path}")
            
        with open(report_path, "r") as f:
            all_reports = json.load(f)
            
        if indices is not None:
            self.samples = [all_reports[i] for i in indices if i < len(all_reports)]
        else:
            self.samples = all_reports

        # Build pathology to class id mapping
        unique_pathologies = sorted(list(set(item["pathology"] for item in all_reports)))
        self.pathology_to_label = {p: i for i, p in enumerate(unique_pathologies)}

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, volume_rel_path, expected_dims):
        volume_path = os.path.join(self.data_dir, "volumes", volume_rel_path)
        if not os.path.exists(volume_path):
            # Try direct path
            volume_path = os.path.join(self.data_dir, volume_rel_path)
            if not os.path.exists(volume_path):
                raise FileNotFoundError(f"Volume file not found: {volume_path}")
        
        raw_data = np.fromfile(volume_path, dtype=np.float32)
        expected_size = int(np.prod(expected_dims))
        if len(raw_data) != expected_size:
            raise ValueError(
                f"Binary byte mismatch for {volume_rel_path}: "
                f"got {len(raw_data)} floats, expected {expected_size}"
            )
            
        volume = raw_data.reshape(expected_dims)
        
        # Check for NaN / Inf
        if np.isnan(volume).any() or np.isinf(volume).any():
            volume = np.nan_to_num(volume, nan=0.0, posinf=3.0, neginf=-3.0)
            
        return volume

    def _augment_volume(self, volume):
        """
        3D spatial and intensity data augmentation:
        - Random 3D axis flips
        - Random 90-degree rotations along spatial planes
        - Subtle intensity jitter
        """
        # Random flips along D, H, W
        for axis in (0, 1, 2):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis)

        # Random 90-degree rotations in axial plane (H, W)
        k = np.random.randint(0, 4)
        if k > 0:
            volume = np.rot90(volume, k, axes=(1, 2))

        # Intensity jitter: random scale (0.9 to 1.1) and shift (-0.1 to 0.1)
        scale = np.random.uniform(0.9, 1.1)
        shift = np.random.uniform(-0.1, 0.1)
        volume = volume * scale + shift
        
        return volume.copy()

    def __getitem__(self, idx):
        item = self.samples[idx]
        dims = tuple(item.get("volume_dimensions", self.target_shape))
        volume_np = self._load_volume(item["volume_file"], dims)
        
        if self.augment:
            volume_np = self._augment_volume(volume_np)

        # Standardize volume to 1-channel tensor: (1, D, H, W)
        volume_tensor = torch.from_numpy(volume_np).float().unsqueeze(0)
        label = self.pathology_to_label[item["pathology"]]

        return {
            "case_id": item["case_id"],
            "pathology": item["pathology"],
            "label": torch.tensor(label, dtype=torch.long),
            "report": item["clinical_radiology_report"],
            "volume": volume_tensor,
            "volume_file": item["volume_file"]
        }


def get_augmented_dataset(data_dir=".", report_file="radiology_reports.json", 
                           num_augmentations_per_sample=10):
    """
    Creates an augmented dataset by producing multiple transformed variations
    of the base volumes while preserving report correspondence.
    Useful for self-supervised pretraining and contrastive alignment.
    """
    base_ds = VolumeReportDataset(data_dir=data_dir, report_file=report_file, augment=False)
    samples = []
    
    # Add originals first
    for i in range(len(base_ds)):
        samples.append(base_ds[i])
        
    # Generate augmented instances
    aug_ds = VolumeReportDataset(data_dir=data_dir, report_file=report_file, augment=True)
    for _ in range(num_augmentations_per_sample):
        for i in range(len(aug_ds)):
            samples.append(aug_ds[i])
            
    return samples
