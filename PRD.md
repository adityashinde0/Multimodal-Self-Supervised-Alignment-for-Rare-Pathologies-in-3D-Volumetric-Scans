# PRD — PS-007 Multimodal Self-Supervised Alignment for Rare Pathologies in 3D Volumetric Scans

## 1. Problem Definition

### Problem

Rare pulmonary and neurological pathologies are difficult to model because available datasets often contain 3D CT/MRI volumes and radiology reports but lack detailed voxel-level annotations.

Traditional supervised 3D segmentation requires physicians to manually annotate voxels or slices. This creates a major data and scalability bottleneck for rare diseases.

Standard 2D vision-language approaches also fail to fully preserve volumetric spatial relationships because a 3D scan contains information across depth that cannot be represented adequately by independently processing 2D slices.

### Proposed Solution

Build a self-supervised multimodal 3D representation-learning system that:

1. Ingests 3D CT/MRI volumes.
2. Converts the volume into 3D volumetric patches.
3. Masks a large percentage of patches during self-supervised pretraining.
4. Uses a 3D masked autoencoder to learn spatial representations from visible patches.
5. Encodes corresponding radiology reports into text representations.
6. Aligns 3D visual representations with report representations using symmetric contrastive learning.
7. Provides a zero-shot pathology retrieval interface.
8. Measures retrieval performance against a defined supervised baseline.
9. Measures inference VRAM and latency.

The core objective is not voxel-level segmentation. The primary evaluation target is zero-shot rare-pathology retrieval.

---

## 2. Core Value Proposition

The system attempts to learn useful 3D medical representations from paired scans and naturally occurring radiology reports without requiring voxel-level disease annotations.

The intended benefit is to transform weakly structured clinical documentation into supervision for learning spatial 3D representations.

The key architectural idea is:

**3D volume → volumetric patches → masked 3D encoder → visual embedding**

combined with:

**radiology report → text encoder → text embedding**

followed by:

**visual/text embedding alignment → zero-shot pathology retrieval**

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Load the supplied 3D volume data from the PS-007 dataset package. |
| FR-02 | Load corresponding radiology reports from `radiology_reports.json`. |
| FR-03 | Validate volume/report correspondence. |
| FR-04 | Convert valid volumes into 3D patch tokens. |
| FR-05 | Apply masked-volume pretraining. |
| FR-06 | Reconstruct masked visual information through the 3D-MAE decoder. |
| FR-07 | Generate a visual embedding for each volume. |
| FR-08 | Generate a text embedding for each corresponding report. |
| FR-09 | Train visual/text representations with symmetric contrastive alignment. |
| FR-10 | Support pathology-text queries for zero-shot retrieval. |
| FR-11 | Rank candidate scans against a textual pathology query. |
| FR-12 | Evaluate retrieval using mean Average Precision (mAP). |
| FR-13 | Evaluate inference GPU memory usage. |
| FR-14 | Provide reproducible evaluation results. |
| FR-15 | Provide a demonstrable end-to-end workflow. |

### 3.2 Non-Functional Requirements

- Reproducible preprocessing.
- Deterministic evaluation configuration where practical.
- Clear train/validation/test separation.
- No test-set leakage through reports or pathology labels.
- GPU-memory-aware architecture.
- Graceful handling of malformed volumes/reports.
- Modular model components.
- Measurable performance rather than unsupported claims.
- Local/reproducible execution wherever possible.

### 3.3 Explicit Constraints

- Mandatory language: Python.
- PyTorch, MONAI and/or JAX are permitted by the problem statement.
- CUDA or Triton is required for the memory-efficient 3D attention direction.
- Target inference memory: ≤24 GB VRAM.
- Target: outperform a standard supervised 3D baseline by at least 15% mAP.
- External public medical datasets may be incorporated.
- Maximum implementation window: 24 hours.

### 3.4 Evaluation Requirements

Primary:

**Zero-shot rare-pathology retrieval mAP**

Secondary:

- Recall@1
- Recall@5
- Recall@10
- Inference peak VRAM
- Inference latency
- Training throughput
- Number of trainable parameters
- Ablation performance

The 15% mAP improvement is an evaluation target, not a pre-existing result.

---

## 4. Users / Actors

### Primary Actor

Radiologist/researcher evaluating rare-pathology retrieval.

### Secondary Actors

- ML engineer
- Medical-imaging researcher
- Hackathon evaluator
- Dataset/evaluation pipeline

### System Actors

- 3D volume loader
- Report loader
- Visual encoder
- Text encoder
- Contrastive alignment module
- Retrieval engine
- Evaluation engine
- Benchmarking/memory profiler

---

## 5. Assumptions

### A-01 — Volume Shape

The exact dimensions of the supplied `.bin` volumes must be verified from `PS-007.zip` and `DATASET_INFO.md`.

The phrase `16x16x16` is currently ambiguous.

It may describe voxel-array dimensions, patch dimensions, or a dataset representation detail.

The final patch tokenizer must not be fixed until the actual volume shape is inspected.

### A-02 — Patch Size

The intended architecture uses approximately `16×16×16` volumetric patches as suggested by the problem statement.

If the actual volume dimensions are too small for this configuration, the patch size must be adapted.

### A-03 — Text Encoder

A pretrained or lightweight clinical-language encoder may be used if compatible with the dataset and available compute.

The exact model must be selected after checking licensing, model availability and hardware requirements.

### A-04 — Dataset Size

The supplied dataset is assumed to be sufficiently large for demonstrating the pipeline, but its actual size must be measured before choosing model capacity.

### A-05 — External Dataset

External datasets are optional.

They will only be added if they materially improve representation learning or evaluation and can be incorporated without introducing unacceptable preprocessing complexity or leakage.

### A-06 — Baseline

The supervised baseline must be explicitly implemented or reproduced using the same evaluation split and comparable preprocessing.

The baseline architecture must be selected after inspecting the dataset.

### A-07 — 15% Criterion

"15% improvement" is interpreted initially as a target relative improvement unless the official judging documentation defines it differently.

The final interpretation must be recorded before reporting results.

---

## 6. MVP Scope

### MVP-01 — Dataset Ingestion

**Purpose:** Load and validate scans and reports.

**Requirement satisfied:** FR-01, FR-02, FR-03.

**Judging value:** Demonstrates reliable medical-data handling.

**Owner:** Programmer 2.

---

### MVP-02 — 3D Patchification

**Purpose:** Convert validated volumes into volumetric tokens.

**Requirement satisfied:** FR-04.

**Judging value:** Demonstrates preservation of 3D spatial structure.

**Owner:** Programmer 1.

---

### MVP-03 — 3D Masked Autoencoder

**Purpose:** Learn spatial representations using masked-volume reconstruction.

**Requirement satisfied:** FR-05, FR-06.

**Judging value:** Core self-supervised learning contribution.

**Owner:** Programmer 1.

---

### MVP-04 — Report Encoder

**Purpose:** Convert free-text reports into semantic representations.

**Requirement satisfied:** FR-08.

**Judging value:** Enables multimodal supervision.

**Owner:** Programmer 2.

---

### MVP-05 — Multimodal Contrastive Alignment

**Purpose:** Align scan and report embeddings using symmetric contrastive learning.

**Requirement satisfied:** FR-09.

**Judging value:** Core multimodal innovation.

**Owner:** Programmer 1.

---

### MVP-06 — Memory-Efficient Attention

**Purpose:** Reduce attention memory requirements sufficiently to make 3D processing practical.

**Requirement satisfied:** Problem's CUDA/Triton direction.

**Judging value:** Addresses the principal systems bottleneck.

**Owner:** Programmer 1 + Programmer 3.

---

### MVP-07 — Zero-Shot Retrieval

**Purpose:** Retrieve scans matching textual pathology descriptions.

**Requirement satisfied:** FR-10, FR-11.

**Judging value:** Direct demonstration of the model's multimodal capability.

**Owner:** Programmer 2.

---

### MVP-08 — Evaluation and Benchmarking

**Purpose:** Measure mAP, Recall@K and VRAM.

**Requirement satisfied:** FR-12, FR-13, FR-14.

**Judging value:** Makes performance claims defensible.

**Owner:** Programmer 3.

---

### MVP-09 — Demonstration Interface

**Purpose:** Provide a simple workflow for uploading/selecting a scan, entering a pathology query and viewing ranked results.

**Requirement satisfied:** FR-15.

**Judging value:** Makes the technical system understandable to evaluators.

**Owner:** Programmer 3.

---

## 7. Out of Scope

The MVP will not prioritize:

- Clinical diagnosis.
- Clinical treatment recommendations.
- Automated physician replacement.
- Production hospital deployment.
- Full voxel-level segmentation.
- Large-scale distributed training.
- Kubernetes/cloud orchestration.
- Complex authentication.
- Real-time hospital integration.
- Large-scale model serving infrastructure.
- Unnecessary external APIs.
- Multiple model families unless required for benchmarking.

These features do not directly improve the core hackathon objective.

---

## 8. Success Metrics

### Primary Metric

Zero-shot rare-pathology retrieval mAP.

Target:

**Proposed model mAP ≥ baseline mAP × 1.15**

provided that this is confirmed as the intended interpretation of the competition criterion.

### Secondary Metrics

- Recall@1
- Recall@5
- Recall@10
- Peak inference VRAM
- Inference latency
- Parameter count
- Training throughput

### Memory Requirement

Peak forward-pass inference VRAM must be measured.

Target:

**Peak VRAM ≤ 24 GB**

This must be measured rather than assumed.

---

## 9. Risks

### R-01 — Insufficient Dataset Size

Small datasets may cause unstable multimodal training.

**Mitigation:** smaller model, stronger regularization, augmentation, external data if justified, and careful split strategy.

### R-02 — Ambiguous Volume Dimensions

Incorrect patchification could destroy useful spatial information.

**Mitigation:** inspect actual binary shape and metadata before model implementation.

### R-03 — Report Noise

Radiology reports may contain incidental findings, negations and irrelevant information.

**Mitigation:** preserve raw reports, optionally create controlled pathology query templates, and evaluate both report-to-volume and pathology-query retrieval.

### R-04 — False Negative Contrastive Pairs

Two scans may contain the same pathology but only one is treated as the positive pair.

**Mitigation:** evaluate class-aware retrieval separately and investigate duplicate/pathology-aware negatives.

### R-05 — GPU Memory Exhaustion

3D token counts can grow rapidly.

**Mitigation:** high masking ratio, compact encoder, mixed precision, memory-efficient attention, profiling, and potentially sparse/local attention.

### R-06 — Failure to Reach 15% Target

The architecture cannot guarantee the target before experiments.

**Mitigation:** establish baseline early, perform ablations, and report the measured result honestly.

### R-07 — Data Leakage

Reports or pathology labels could accidentally leak test information.

**Mitigation:** split by patient/study where identifiers permit and construct preprocessing only from training data.

---

## 10. Demo Strategy

The shortest compelling demonstration:

1. Select a 3D CT/MRI volume.
2. Show its volumetric dimensions and preprocessing.
3. Display the model's learned visual embedding.
4. Enter a textual pathology query such as a rare pathology description.
5. Run zero-shot retrieval.
6. Display the top-ranked scans.
7. Show retrieval scores.
8. Compare proposed model mAP against supervised baseline.
9. Display peak VRAM.
10. Explain how the system learns from scans + reports without voxel-level annotations.

The demo should visually emphasize:

**3D scan + report → shared representation → pathology query → ranked scans**

---

## Evidence / Decision Notes

### Decision: 3D MAE

**Evidence:** The original MAE paper demonstrates masked patch reconstruction and reports that high masking ratios such as 75% can be effective for self-supervised visual representation learning.

**Reason:** This directly supports the self-supervised portion of the proposed architecture.

**Confidence:** High for the general MAE mechanism; medium for its effectiveness on this specific medical dataset.

### Decision: Contrastive Vision/Text Alignment

**Evidence:** CLIP demonstrated learning aligned image/text representations using paired image-text data and contrastive learning.

**Reason:** The PS-007 task explicitly requires scan/report alignment.

**Confidence:** High for the general method; performance on PS-007 remains empirical.

### Decision: PyTorch SDPA / Efficient Attention

**Evidence:** Current PyTorch documentation provides scaled-dot-product attention implementations including FlashAttention-2 and memory-efficient attention.

**Reason:** These provide an existing optimized path before writing a custom kernel.

**Confidence:** High.

### Decision: Triton

**Evidence:** Triton's official tutorials include fused-attention implementations.

**Reason:** Triton can be used if profiling demonstrates that the required 3D attention pattern needs a custom kernel.

**Confidence:** High for kernel-development capability; actual PS-007 benefit requires benchmarking.

### Decision: JAX

**Evidence:** JAX provides accelerator-oriented array computation and explicit GPU-memory management mechanisms.

**Reason:** Useful for research experimentation, but introducing a second primary ML framework unnecessarily increases implementation complexity.

**Confidence:** High.

---

## Final Product Definition

The MVP is a:

> **Self-supervised 3D medical vision-language retrieval system that learns spatial representations from unlabeled 3D scans and paired radiology reports, then performs zero-shot rare-pathology retrieval under a 24 GB inference-memory constraint.**

The system does not claim clinical diagnosis or guaranteed superiority until experimental evaluation demonstrates it.