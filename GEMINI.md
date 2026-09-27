# GEMINI.md - Agent Project Guide & Operational Instructions

> **Purpose**: This file provides immediate context, architectural invariants, environment specifications, testing protocols, and Git guidelines for AI agents working in this repository to prevent redundant token consumption and prevent context rediscovery.

---

## 1. Project Overview & Core Concept

* **Project Name**: PS-007: Multimodal Self-Supervised Alignment for Rare Pathologies in 3D Volumetric Scans
* **Repository**: [`adityashinde0/Multimodal-Self-Supervised-Alignment-for-Rare-Pathologies-in-3D-Volumetric-Scans`](https://github.com/adityashinde0/Multimodal-Self-Supervised-Alignment-for-Rare-Pathologies-in-3D-Volumetric-Scans.git)
* **Core Problem**: Rare anatomical pathologies in 3D volumetric scans (CT/MRI) lack dense voxel annotations. Supervised models fail to generalize.
* **Solution**:
  1. **Self-Supervised 3D Masked Autoencoder (3D-MAE)**: Learns anatomical spatial continuity from raw 3D scans without manual labels using 75% volumetric masking.
  2. **Multimodal Alignment**: Projects 3D anatomical patch representations and clinical text report embeddings (via frozen HuggingFace transformer) into a shared **128-D L2-normalized hypersphere** optimized with symmetric InfoNCE contrastive loss.
  3. **Zero-Shot Retrieval**: Enables radiologist natural-language queries (e.g., *"heterogeneous hyperdense lesion with peripheral ring enhancement"*) to retrieve the exact 3D volume without fine-tuning.

---

## 2. Environment & Execution Setup

### Python Virtual Environment
Always use the dedicated Python virtual environment:
* **Interpreter Path**: `/home/shindeadi/.venvs/ps007/bin/python`
* **Activation**:
  ```bash
  source ~/.venvs/ps007/bin/activate
  ```

### Hardware Environment
* **Primary GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0 / PyTorch 2.x)
* **Device Fallback**: All modules support seamless dynamic fallback (`"cuda" if torch.cuda.is_available() else "cpu"`).
* **VRAM Budget**: Optimized for < 4 GB VRAM (peak observed inference footprint is ~122.54 MB).

---

## 3. Fast Verification & Testing Commands

Before making commits or proposing changes, **always execute the verification suite**:

```bash
# 1-Click System Audit & Jury Scorecard (<10s runtime)
/home/shindeadi/.venvs/ps007/bin/python verify_all.py

# Run all 15 automated unit tests via Pytest
/home/shindeadi/.venvs/ps007/bin/pytest tests/test_pipeline.py -v

# Or run via standard unittest
/home/shindeadi/.venvs/ps007/bin/python -m unittest tests/test_pipeline.py -v
```

---

## 4. Key Architectural & Data Invariants

| Component | Invariant Specification |
|---|---|
| **Input Volume Shape** | `(B, 1, 16, 16, 16)` single-channel 3D volume |
| **Patch Size** | `(4, 4, 4)` voxels |
| **Token Count** | $(16/4)^3 = 64$ volumetric tokens |
| **Masking Ratio** | **75%** exactly (48 masked tokens, 16 visible tokens) |
| **Latent Dimension** | 128-D projection |
| **Embedding Space** | Unit hypersphere ($\ell_2$-normalized: `torch.norm(embed, dim=-1) == 1.0`) |
| **Text Encoder** | `sentence-transformers/all-MiniLM-L6-v2` (frozen, 384-D -> 128-D linear projector) |
| **Loss Function** | Symmetric InfoNCE contrastive loss with learnable temperature $\tau$ |

---

## 5. Repository File Map

```text
PS-007-GT/
├── GEMINI.md                                  # You are here: agent memory & instructions
├── DEFENSE_FAQ.md                             # Hackathon jury defense Q&A & methodology proof
├── ARCHITECTURE.md                            # Comprehensive mathematical & system architecture
├── PRD.md                                     # Product Requirements Document
├── PROGRESS.md                                # Development history & milestone tracker
├── README.md                                  # Public repository documentation
├── verify_all.py                              # 1-click jury audit & validation script
├── pytest.ini                                 # Pytest configuration (pythonpath = .)
├── src/
│   ├── data/dataset.py                        # 3D dataset, splits, and synthetic volumes
│   ├── eval/
│   │   ├── metrics.py                         # mAP, Recall@K, LOCO cross-validation
│   │   ├── profiler.py                        # GPU VRAM & latency profiler
│   │   └── retrieval.py                       # Cosine similarity ranking & search gallery
│   ├── models/
│   │   ├── aligner.py                         # MultimodalAligner & InfoNCELoss
│   │   ├── attention.py                       # Multi-head self-attention blocks
│   │   ├── baseline.py                        # Supervised baseline comparison model
│   │   ├── mae3d.py                           # 3D-MAE encoder & decoder
│   │   ├── patch_embed.py                     # 3D patch embedding & sinusoidal pos-embed
│   │   ├── projector.py                       # Dimension projection heads
│   │   └── text_encoder.py                    # HuggingFace transformer wrapper
│   ├── train.py                               # Training pipeline (MAE pretraining & aligner)
│   └── utils.py                               # Seed setting, device utils, tensor helpers
├── tests/
│   └── test_pipeline.py                       # 15 comprehensive unit tests
├── artifacts/
│   ├── checkpoints/                           # Trained PyTorch model weights (.pt)
│   └── metrics/benchmark_results.json         # Authoritative empirical benchmark results
└── frontend/                                  # Interactive Radiology Light Box Web Application
    ├── index.html
    ├── css/styles.css
    └── js/
        ├── app.js
        └── data.js
```

---

## 6. Git & GitHub Push Protocol

* **Target Account**: `adityashinde0` (Aditya Shinde, `adityacsmu007@gmail.com`)
* **Credential Manager**: WSL is configured to use Windows Git Credential Manager (`git-credential-manager.exe`). All pushes authenticate automatically as `adityashinde0`.
* **Current Working Branch**: `audit-fix-scientific-reproducibility`
* **Remote**: `origin` (`https://github.com/adityashinde0/Multimodal-Self-Supervised-Alignment-for-Rare-Pathologies-in-3D-Volumetric-Scans.git`)
* **Push Procedure**:
  ```bash
  git add <files>
  git commit -m "feat/fix: descriptive message"
  git push origin audit-fix-scientific-reproducibility
  ```

---

## 7. Guidelines for Agents & Contributors

1. **No Synthetic / Mocked Claims**: Never inject fabricated benchmark numbers into docs or code. All metrics must be backed by `artifacts/metrics/benchmark_results.json` and generated by running `verify_all.py`.
2. **Preserve Determinism**: Maintain `set_seed(42)` across tests and experiments to ensure reproducible rankings.
3. **Keep Imports Relative to Root**: Modules import via `from src.xxx import ...`. Keep `pytest.ini` with `pythonpath = .` intact.
4. **Conserve Tokens**: Refer directly to `GEMINI.md`, `DEFENSE_FAQ.md`, and `benchmark_results.json` instead of executing broad file searches across the entire workspace.
