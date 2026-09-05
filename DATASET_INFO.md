# PS-007: 3D Volumetric Scans & Radiology Reports
Contains 3D volumetric medical scans paired with unstructured clinical radiology free-text reports.
- `volumes/`: 3D voxel arrays (.bin raw float32 format, 16x16x16) representing CT/MRI scans with localized rare pathology signatures.
- `radiology_reports.json`: Corresponding radiologist free-text reports for multimodal self-supervised alignment (3D-MAE + InfoNCE loss) and zero-shot retrieval.
