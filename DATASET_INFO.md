# 🩺 PS-007 Dataset Specifications & Extensibility Guide

## 1. Dataset Overview
This dataset contains paired 3D volumetric medical scans (CT and MRI) and unstructured narrative radiology reports designed for multimodal self-supervised representation alignment.

- **Primary Challenge**: Learning 3D spatial representations for rare medical pathologies where voxel-level manual segmentations do not exist.
- **Dataset Nature**: Proof-of-Concept (PoC) Feasibility Prototype across 5 curated rare medical conditions (1 volumetric scan per pathology class).
- **Voxel Representation**: Raw IEEE 754 float32 continuous intensity grids without compression artifacts.

---

## 2. Case Registry

| Case ID | Condition | Modality / Anatomy | Binary File | Key Diagnostic Hallmarks |
|---|---|---|---|---|
| **`CASE_000`** | **Lymphangioleiomyomatosis (LAM)** | High-Resolution Chest CT | `volume_00_lymphangioleiomyomatosis.bin` | Thin-walled, uniform round air cysts with preserved lung volume. |
| **`CASE_001`** | **Idiopathic Pulmonary Fibrosis (IPF)** | High-Resolution Chest CT | `volume_01_idiopathic_pulmonary_fibrosis.bin` | Subpleural basilar reticular opacities, honeycombing, traction bronchiectasis. |
| **`CASE_002`** | **Glioblastoma Multiforme (GBM)** | Brain MRI (T1-Gd / FLAIR) | `volume_02_glioblastoma_multiforme.bin` | Heterogeneous rim-enhancing necrotic frontal mass with vasogenic edema. |
| **`CASE_003`** | **Pulmonary Alveolar Proteinosis (PAP)** | High-Resolution Chest CT | `volume_03_pulmonary_alveolar_proteinosis.bin` | Bilateral symmetric ground-glass opacities with crazy-paving septal thickening. |
| **`CASE_004`** | **Creutzfeldt-Jakob Disease (CJD)** | Brain MRI (DWI / FLAIR) | `volume_04_creutzfeldt-jakob_disease.bin` | Cortical ribboning and bilateral symmetric caudate/putamen hyperintensity. |

---

## 3. Data Format & Technical Specifications

- **Spatial Dimensions**: $16 \times 16 \times 16$ voxels ($D \times H \times W$).
- **Channels**: 1 (monochrome volumetric scalar field).
- **Total Voxels per Volume**: $16^3 = 4,096$ voxels.
- **Byte Footprint per Volume**: $4,096 \times 4\text{ bytes} = 16,384\text{ bytes}$ ($16\text{ KB}$).
- **Normalization**: Zero-mean, unit-variance standardized float32 arrays with NaN/Inf sanitization.

---

## 4. Scalability & Extensibility Guide (Adding New Cases)

The data pipeline in `src/data/dataset.py` is dynamically parameterized and scalable. **Adding new cases requires zero code changes.**

To add a new volumetric scan:
1. Generate or export your raw volumetric array as float32 binary:
   ```python
   import numpy as np
   volume = np.random.randn(16, 16, 16).astype(np.float32)
   volume.tofile("volumes/volume_05_sarcoidosis.bin")
   ```
2. Append a new case metadata block to `radiology_reports.json`:
   ```json
   {
     "case_id": "CASE_005",
     "pathology": "Pulmonary Sarcoidosis",
     "volume_dimensions": [16, 16, 16],
     "volume_file": "volume_05_sarcoidosis.bin",
     "clinical_radiology_report": "Bilateral hilar and mediastinal lymphadenopathy with perilymphatic nodular opacities along bronchovascular bundles."
   }
   ```
3. Re-run `run_experiments.py` or restart `server.py`. The `VolumeReportDataset` automatically parses the new entry, updates the class registry, and re-indexes the 3D gallery.

---

## 5. Scientific Limitations & Honest Scope Statement

> [!WARNING]
> **Extreme Low-Data Feasibility Scope**:
> - With only $N=5$ distinct pathologies in this benchmark dataset, i.i.d. train/val/test splits with multi-patient variance are statistically impossible.
> - The evaluation benchmark measures representation alignment across the 5 indexed challenge cases.
> - In Leave-One-Case-Out (LOCO) cross-validation where an entire rare disease is held out from training, mean Recall@1 is $0.2000$, correctly reflecting the difficulty of extreme zero-shot generalization without seeing any pathology examples.
> - **This system is an engineering and algorithmic proof-of-concept. It does NOT claim clinical diagnostic validation, clinical certification, or generalizability to arbitrary hospital populations.**
