# ARCHITECTURE — PS-007 Multimodal Self-Supervised Alignment for Rare Pathologies

## 1. Architecture Overview

The proposed architecture is a multimodal self-supervised learning pipeline.

### High-Level Architecture

```text
                 ┌─────────────────────┐
                 │   3D CT / MRI Scan  │
                 └──────────┬──────────┘
                            │
                     Volume Validation
                            │
                     Normalization
                            │
                    3D Patchification
                            │
                  Random Masking (~75%)
                            │
                            ▼
                 ┌─────────────────────┐
                 │   3D MAE Encoder    │
                 │ Visible Patch Only  │
                 └──────────┬──────────┘
                            │
                     Visual Tokens
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Multimodal Projector│
                 └──────────┬──────────┘
                            │
                      Image Embedding
                            │
                            │
                            │ Symmetric
                            │ Contrastive
                            │ Alignment
                            ▼
                 ┌─────────────────────┐
                 │ Shared Embedding    │
                 │      Space          │
                 └─────────────────────┘
                            ▲
                            │
                      Text Embedding
                            │
                 ┌──────────┴──────────┐
                 │ Text Projector      │
                 └──────────▲──────────┘
                            │
                 ┌─────────────────────┐
                 │ Clinical Text       │
                 │ Encoder             │
                 └──────────▲──────────┘
                            │
                 Radiology Report
```

### Retrieval Flow

```text
Text Pathology Query
        │
        ▼
Clinical Text Encoder
        │
        ▼
Text Embedding
        │
        ▼
Similarity Search
        │
        ▼
Stored Scan Embeddings
        │
        ▼
Ranked Scan Results
        │
        ├──────────────► Recall@K
        │
        └──────────────► mAP
```

---

## 2. System Flow

### Training

```text
Dataset
   │
   ├───────────────► 3D Volume
   │                    │
   │                    ▼
   │             Validate / Normalize
   │                    │
   │                    ▼
   │              Patchify Volume
   │                    │
   │                    ▼
   │              Mask ~75%
   │                    │
   │                    ▼
   │              3D MAE Encoder
   │                    │
   │                    ▼
   │              Visual Embedding
   │                    │
   │                    ▼
   │             Projection Head
   │                    │
   │                    ├─────────────┐
   │                                  │
   │                                  ▼
   │                         Contrastive Loss
   │                                  ▲
   │                                  │
   │                    ┌─────────────┘
   │                    │
   └──────► Report ──► Text Encoder
                         │
                         ▼
                   Text Embedding
                         │
                         ▼
                   Projection Head
```

### Failure Path

```text
Input
 │
 ▼
Validation
 │
 ├── invalid ──► reject + log reason
 │
 ▼
Preprocessing
 │
 ├── preprocessing failure ──► skip sample + log
 │
 ▼
Model Inference
 │
 ├── OOM ──► reduced inference configuration / safe failure
 │
 ▼
Embedding
 │
 ├── invalid embedding ──► reject result + log
 │
 ▼
Retrieval
 │
 └── no valid candidates ──► return explicit empty result
```

---

## 3. Component Architecture

## 3.1 Dataset Ingestion

### Responsibility

Load raw volumes and corresponding reports.

### Inputs

- `.bin` volume files
- `radiology_reports.json`
- `DATASET_INFO.md`

### Outputs

Validated sample:

```text
Sample {
    volume
    report
    sample_id
    metadata
}
```

### Dependencies

- Python
- NumPy
- MONAI where useful

---

## 3.2 Volume Validation

### Responsibility

Verify:

- File exists.
- Binary length is valid.
- Float32 interpretation is correct.
- Expected dimensions are known.
- No unexpected NaN/Inf values.
- Report correspondence exists.

### Output

Validated tensor:

```text
[B, C, D, H, W]
```

The exact dimensions must be determined from the supplied dataset.

---

## 3.3 Preprocessing

Potential operations:

- dtype conversion
- intensity normalization
- clipping where medically justified
- spatial normalization where metadata supports it
- augmentation during training

Preprocessing must not be invented from generic medical-imaging assumptions.

The actual dataset schema must determine which transformations are appropriate.

---

## 3.4 3D Patchification

The volume is divided into volumetric patches.

Target example:

```text
Patch = 16 × 16 × 16
```

Each patch becomes a token.

### Important Validation

If the entire supplied volume is only `16×16×16`, then this patch size produces one token and cannot support meaningful 75% patch masking.

Therefore:

**Actual volume dimensions must be inspected before finalizing this component.**

---

## 3.5 Masking Engine

Target:

```text
Mask ratio ≈ 75%
Visible ratio ≈ 25%
```

Only visible patches are passed to the MAE encoder.

The decoder receives encoded visible information plus mask tokens and attempts reconstruction.

The masking ratio is configurable rather than hard-coded so that ablations can be performed.

---

## 3.6 3D MAE Encoder

### Responsibility

Learn spatial representations from visible 3D patches.

### Input

Visible 3D patch tokens.

### Output

Encoded visual tokens.

### Design

Initial MVP:

- Patch embedding
- 3D positional encoding
- Transformer encoder
- Lightweight decoder
- Reconstruction head

The encoder should operate primarily on visible tokens to reduce computational cost.

---

## 3.7 MAE Decoder

### Responsibility

Reconstruct masked information.

### Input

- Visible latent tokens
- Mask tokens
- Positional information

### Output

Reconstructed patch representation.

### Training Signal

Masked reconstruction loss.

---

## 3.8 Clinical Text Encoder

### Responsibility

Convert radiology reports into semantic representations.

### Input

Raw report.

### Output

Text embedding.

### Selection Rule

The exact text model is not fixed before:

1. Dataset inspection.
2. Model availability verification.
3. License verification.
4. Hardware estimation.
5. Baseline experimentation.

---

## 3.9 Projection Heads

Separate projection heads map visual and text representations into a shared embedding space.

```text
Visual Representation
        │
        ▼
Visual Projection
        │
        ▼
Shared Dimension


Text Representation
        │
        ▼
Text Projection
        │
        ▼
Shared Dimension
```

Normalization:

```text
z = normalize(projection(x))
```

---

## 3.10 Symmetric Contrastive Alignment

For a batch containing N paired scan/report samples:

```text
Image Embeddings:
I1, I2, ..., IN

Text Embeddings:
T1, T2, ..., TN
```

Compute similarity:

```text
Sij = cosine(Ii, Tj) / temperature
```

The diagonal represents matched pairs.

The loss contains both directions:

```text
Image → Text
Text → Image
```

Combined:

```text
Lcontrastive =
    (Limage_to_text + Ltext_to_image) / 2
```

The exact weighting between MAE reconstruction loss and contrastive loss must be tuned experimentally.

---

## 3.11 Attention / Memory Strategy

### Primary Strategy

Use PyTorch's optimized scaled-dot-product attention where applicable.

PyTorch currently supports optimized attention implementations including FlashAttention-2 and memory-efficient attention.

### Secondary Strategy

If the required 3D token pattern cannot be handled efficiently enough, implement a targeted Triton kernel.

### Important

Do not write a custom CUDA/Triton kernel before profiling.

The optimization sequence is:

```text
Baseline attention
       ↓
PyTorch SDPA
       ↓
Profile
       ↓
Identify bottleneck
       ↓
Triton/CUDA optimization if justified
       ↓
Benchmark
```

This avoids unnecessary low-level code.

---

## 3.12 Retrieval Engine

### Input

Text pathology query.

Example:

```text
"rare pulmonary pathology with characteristic diffuse cystic involvement"
```

### Processing

```text
Query
 ↓
Text Encoder
 ↓
Projection
 ↓
Normalization
 ↓
Similarity against scan embeddings
 ↓
Ranking
```

### Output

```text
[
    {
        scan_id,
        similarity_score,
        rank
    },
    ...
]
```

---

## 3.13 Evaluation Engine

Measures:

- mAP
- Recall@1
- Recall@5
- Recall@10
- inference latency
- peak VRAM

It must use a fixed evaluation split.

---

## 4. Data / Storage Design

### Dataset Storage

Use the supplied filesystem dataset.

```text
dataset/
├── volumes/
├── radiology_reports.json
└── DATASET_INFO.md
```

### Model Artifacts

```text
artifacts/
├── checkpoints/
├── embeddings/
├── metrics/
└── logs/
```

### Database

**PostgreSQL: Not required for this problem.**

A relational database would add complexity without directly improving the core self-supervised learning or retrieval objective.

### Vector Database

**Vector database: Not required for the MVP.**

The dataset size should first be measured.

For a hackathon-scale evaluation set, an in-memory tensor similarity operation or a simple local index is sufficient.

A dedicated vector database should only be introduced if the measured corpus size makes it necessary.

---

## 5. Core Interfaces

### Dataset Interface

```python
class VolumeReportDataset:
    def __getitem__(self, index):
        return {
            "volume": volume,
            "report": report,
            "sample_id": sample_id,
            "metadata": metadata,
        }
```

### Visual Encoder

```python
class VolumeEncoder:
    def forward(self, volume):
        return visual_embedding
```

### Text Encoder

```python
class ReportEncoder:
    def forward(self, report):
        return text_embedding
```

### Alignment Model

```python
class MultimodalAligner:
    def forward(self, volume, report):
        return {
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "loss": loss,
        }
```

### Retrieval

```python
def retrieve(
    text_query,
    scan_embeddings,
    text_encoder,
    top_k
):
    ...
```

### Evaluation

```python
def evaluate_retrieval(
    query_embeddings,
    scan_embeddings,
    ground_truth
):
    ...
```

---

## 6. Technology Decisions

| Technology | Decision | Reason | Alternative | Alternative Rejected Because |
|---|---|---|---|---|
| Python | Selected | Mandatory language | — | Required |
| PyTorch | Primary | Strong fit for custom model development and GPU execution | JAX | JAX would add framework complexity without clear MVP benefit |
| MONAI | Selected where useful | Medical-imaging preprocessing/data utilities | Pure PyTorch | More manual medical-data handling |
| JAX | Optional | Useful for research/benchmark experiments | PyTorch only | PyTorch remains simpler for MVP |
| CUDA/Triton | Conditional | Low-level optimization if profiling identifies a bottleneck | PyTorch SDPA only | Custom kernels may be unnecessary |
| PostgreSQL | Rejected | No core requirement | SQLite/PostgreSQL | Persistent relational storage adds unnecessary complexity |
| Vector DB | Rejected initially | Dataset size may not justify it | FAISS/local tensor search | Simpler for MVP |
| Web frontend | Optional | Useful for demonstration | CLI | CLI may be sufficient if time is limited |

---

## 7. Security / Reliability

This is a research prototype rather than a clinical deployment.

Relevant safeguards:

- Never expose patient-identifying metadata unnecessarily.
- Do not treat model output as a clinical diagnosis.
- Validate input files before tensor construction.
- Reject malformed binary files.
- Prevent unbounded input dimensions.
- Catch GPU OOM conditions.
- Log failed samples.
- Keep evaluation data isolated from training.
- Avoid test-label leakage.

---

## 8. Performance Strategy

### Primary Optimization Target

Inference memory.

Target:

```text
Peak VRAM <= 24 GB
```

### Measurements

For every evaluation configuration record:

```text
Model
Input shape
Batch size
Precision
Token count
Peak VRAM
Latency
mAP
Recall@K
```

### Optimization Sequence

```text
1. Establish functional baseline
2. Measure VRAM
3. Reduce unnecessary token computation
4. Enable mixed precision
5. Use optimized attention
6. Profile
7. Implement custom kernel only if justified
8. Re-measure
```

No performance improvement should be claimed without measurements.

---

## 9. Failure & Fallback Strategy

| Failure | Detection | Fallback |
|---|---|---|
| Missing volume | File validation | Skip sample + log |
| Invalid binary size | Byte-count validation | Reject sample |
| NaN/Inf values | Tensor validation | Reject or sanitize according to dataset policy |
| Missing report | Dataset consistency check | Skip unmatched pair |
| Text encoding failure | Encoder exception | Log and skip sample |
| GPU OOM | CUDA exception/memory monitoring | Reduce batch/precision or use safe inference configuration |
| Attention kernel unavailable | Runtime capability check | PyTorch fallback implementation |
| Invalid embedding | NaN/Inf check | Exclude from retrieval |
| No retrieval candidates | Empty index check | Return explicit empty result |
| Poor baseline performance | Evaluation | Revisit preprocessing/model capacity |
| 15% target not reached | Benchmark | Report actual result; do not fabricate improvement |

---

## 10. Engineering Invariants

The following properties must remain true:

1. Every training pair contains a valid volume and corresponding report.
2. Volume tensor dimensions match the configured model input contract.
3. Patchification is reversible or reconstructable for the MAE pipeline.
4. Masked patches are not accidentally provided to the encoder as visible information.
5. Positive image/text pairs correspond to the same sample.
6. Evaluation samples are not used for model fitting.
7. Retrieval embeddings use the same embedding-space contract.
8. Similarity ranking is deterministic under the evaluation configuration where practical.
9. Peak inference VRAM is measured rather than inferred.
10. Model output is not represented as a clinical diagnosis.
11. No invalid tensor values reach the contrastive loss.
12. No unsupported benchmark claim is included in the final presentation.

---

## 11. Technical Trade-offs

### 3D MAE vs Supervised Training

**Chosen:** 3D MAE.

Reason:

The problem specifically targets learning from scans and reports without requiring dense voxel annotations.

Trade-off:

Self-supervised representation learning introduces optimization complexity and does not guarantee superior downstream performance.

---

### 75% Masking vs Lower Mask Ratio

**Initial:** approximately 75%.

Reason:

The problem explicitly suggests 75%, and the original MAE work provides evidence that high masking ratios can be effective.

Trade-off:

The optimal ratio for medical 3D volumes is not guaranteed to be 75%.

Therefore:

```text
75% = initial configuration
not
75% = experimentally proven optimum
```

---

### Dense Attention vs Memory-Efficient Attention

**Chosen:** optimized/fused attention first.

Reason:

Dense attention can create significant memory pressure as token count increases.

Trade-off:

Specialized kernels introduce hardware and implementation complexity.

---

### Custom Triton Kernel vs Existing PyTorch Kernels

**Chosen:** existing PyTorch optimized kernels first.

Reason:

The existing implementation provides a lower-risk optimization path.

Triton becomes justified only when profiling identifies an attention bottleneck that materially affects the target.

---

### Database vs Filesystem

**Chosen:** filesystem.

Reason:

The core task is model training and retrieval rather than transactional application data.

---

## 12. Accepted Technical Debt

The following are acceptable for the 24-hour MVP:

- Small model rather than a large foundation model.
- Limited external datasets.
- Simple local retrieval index.
- Minimal demonstration UI.
- Limited hyperparameter search.
- No distributed training.
- No production serving cluster.
- No full clinical deployment layer.
- No comprehensive clinical validation.

These limitations must be stated explicitly.

---

## 13. Evaluation Protocol

### Step 1 — Dataset Split

Create:

```text
Train
Validation
Test
```

Prefer patient-level separation when patient identifiers are available.

### Step 2 — Supervised Baseline

Train a standard supervised 3D baseline on the available labeled pathology information.

Record:

```text
Baseline mAP
Baseline Recall@K
Baseline VRAM
```

### Step 3 — Proposed Model

Train:

```text
3D MAE + text encoder + contrastive alignment
```

Record the same metrics.

### Step 4 — Compare

```text
Relative improvement =
(Proposed mAP - Baseline mAP) / Baseline mAP
```

If the competition defines 15% as an absolute percentage-point increase rather than relative improvement, use that definition instead.

### Step 5 — Ablations

At minimum:

```text
A. Supervised baseline
B. 3D MAE only
C. MAE + text alignment
D. MAE + text alignment + optimized attention
```

This demonstrates which architectural components contribute to performance.

---

## 14. Evidence Register

### E-01 — Masked Autoencoders

Source:

He et al., *Masked Autoencoders Are Scalable Vision Learners*, 2021.

Supports:

- Masked patch reconstruction.
- Asymmetric encoder/decoder concept.
- High masking ratios such as 75%.

Confidence:

**High for the underlying method; medium for PS-007 performance.**

### E-02 — Image/Text Contrastive Learning

Source:

Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021.

Supports:

- Paired visual/text representation learning.
- Contrastive image/text alignment.
- Zero-shot text-to-image retrieval/classification concept.

Confidence:

**High for the general mechanism; PS-007 effectiveness remains empirical.**

### E-03 — PyTorch Attention

Source:

Official PyTorch scaled-dot-product attention documentation.

Supports:

- Optimized scaled-dot-product attention.
- FlashAttention-2 backend.
- Memory-efficient attention backend.
- Runtime backend selection.

Confidence:

**High.**

### E-04 — Triton Attention

Source:

Official Triton fused-attention tutorial.

Supports:

- Implementing fused attention kernels with Triton.

Confidence:

**High for implementation capability; PS-007 performance remains to be benchmarked.**

### E-05 — JAX

Source:

Official JAX documentation.

Supports:

- Accelerator-oriented computation.
- GPU execution.
- Memory allocation controls.

Confidence:

**High.**

---

## 15. Architecture Decision Summary

The MVP architecture is:

```text
                    ┌────────────────────┐
                    │ 3D CT/MRI Volume   │
                    └─────────┬──────────┘
                              │
                     Normalize/Patchify
                              │
                        75% Masking
                              │
                              ▼
                    ┌────────────────────┐
                    │ 3D MAE Encoder     │
                    └─────────┬──────────┘
                              │
                       Visual Projection
                              │
                              ▼
                       Image Embedding
                              │
                              │
                       Contrastive Loss
                              │
                              ▲
                              │
                       Text Embedding
                              ▲
                              │
                     Text Projection
                              ▲
                              │
                    ┌────────────────────┐
                    │ Clinical Text      │
                    │ Encoder             │
                    └─────────▲──────────┘
                              │
                       Radiology Report


After Training:

Pathology Query
      │
      ▼
Text Encoder
      │
      ▼
Text Embedding
      │
      ▼
Similarity Search
      │
      ▼
Ranked 3D Scans
      │
      ▼
mAP / Recall@K
```

The architecture deliberately avoids unnecessary databases, microservices, agents, cloud infrastructure and unrelated AI components.