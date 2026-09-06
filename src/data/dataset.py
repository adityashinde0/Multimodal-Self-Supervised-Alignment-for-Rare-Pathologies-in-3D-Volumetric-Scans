import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class VolumeReportDataset(Dataset):
    """
    Dataset loader for paired 3D volumetric scans (.bin raw float32)
    and unstructured clinical radiology reports.
    
    Scalability Note:
    - Adding new cases requires only adding entries to `radiology_reports.json`
      and placing the corresponding .bin volumes in the `volumes/` directory.
      Volume dimensions are read directly from the JSON metadata.
    """
    def __init__(
        self, 
        data_dir=".", 
        report_file="radiology_reports.json", 
        augment=False, 
        target_shape=(16, 16, 16), 
        indices=None,
        case_ids=None,
        enable_flips=True,
        enable_rotations=True,
        enable_intensity_jitter=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.augment = augment
        self.enable_flips = enable_flips
        self.enable_rotations = enable_rotations
        self.enable_intensity_jitter = enable_intensity_jitter
        
        report_path = os.path.join(data_dir, report_file)
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file not found at: {report_path}")
            
        with open(report_path, "r") as f:
            all_reports = json.load(f)

        # Pathology-to-label mapping based on full report registry
        unique_pathologies = sorted(list(set(item["pathology"] for item in all_reports)))
        self.pathology_to_label = {p: i for i, p in enumerate(unique_pathologies)}
            
        # Case-level filtering for strict split separation (zero leakage)
        if case_ids is not None:
            case_id_set = set(case_ids)
            self.samples = [item for item in all_reports if item["case_id"] in case_id_set]
        elif indices is not None:
            self.samples = [all_reports[i] for i in indices if i < len(all_reports)]
        else:
            self.samples = all_reports

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, volume_rel_path, expected_dims):
        volume_path = os.path.join(self.data_dir, "volumes", volume_rel_path)
        if not os.path.exists(volume_path):
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
        
        # Guard against NaN / Inf values
        if np.isnan(volume).any() or np.isinf(volume).any():
            volume = np.nan_to_num(volume, nan=0.0, posinf=3.0, neginf=-3.0)
            
        return volume

    def _augment_volume(self, volume):
        """
        Medical Data Augmentation with documented anatomical assumptions:
        1. Left-Right horizontal reflection (Axis 2, Width):
           - Valid for bilaterally symmetric anatomical structures (lungs, cerebrum).
           - Preserves overall pathology morphology while altering lateral coordinates.
        2. In-plane axial rotation (90 deg in H-W plane):
           - Simulates slight rotational misalignments in scanner bore.
           - Through-plane (z-axis / cranio-caudal) topology is preserved.
        3. Subtle intensity scaling (0.95 - 1.05) & shift (-0.05 - 0.05):
           - Simulates inter-scanner calibration drift and Hounsfield Unit / MR intensity variance.
           - Kept conservative to avoid altering diagnostic contrast ratios.
        
        Note: Augmentation is strictly applied ONLY to training cases and NEVER
        to validation or test cases.
        """
        augmented = volume.copy()

        # 1. Anatomical horizontal flip (Left-Right reflection along width)
        if self.enable_flips and np.random.rand() > 0.5:
            augmented = np.flip(augmented, axis=2)

        # 2. In-plane axial rotation (90-degree increments in axial slice plane)
        if self.enable_rotations:
            k = np.random.randint(0, 4)
            if k > 0:
                augmented = np.rot90(augmented, k, axes=(1, 2))

        # 3. Subtle intensity jitter
        if self.enable_intensity_jitter:
            scale = np.random.uniform(0.95, 1.05)
            shift = np.random.uniform(-0.05, 0.05)
            augmented = augmented * scale + shift
        
        return augmented.copy()

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


def get_augmented_dataset(
    data_dir=".", 
    report_file="radiology_reports.json", 
    num_augmentations_per_sample=10,
    indices=None,
    case_ids=None,
    enable_flips=True,
    enable_rotations=True,
    enable_intensity_jitter=True
):
    """
    Creates an augmented dataset strictly for training cases.
    Ensures that test or validation cases are never augmented or leaked into training.
    """
    base_ds = VolumeReportDataset(
        data_dir=data_dir, 
        report_file=report_file, 
        augment=False,
        indices=indices,
        case_ids=case_ids
    )
    samples = []
    
    # 1. Add unaugmented base cases from the designated training split
    for i in range(len(base_ds)):
        samples.append(base_ds[i])
        
    # 2. Generate augmented variations exclusively for these training cases
    aug_ds = VolumeReportDataset(
        data_dir=data_dir, 
        report_file=report_file, 
        augment=True,
        indices=indices,
        case_ids=case_ids,
        enable_flips=enable_flips,
        enable_rotations=enable_rotations,
        enable_intensity_jitter=enable_intensity_jitter
    )
    for _ in range(num_augmentations_per_sample):
        for i in range(len(aug_ds)):
            samples.append(aug_ds[i])
            
    return samples
