import os
import sys
import warnings

# Suppress third-party TensorFlow/Protobuf/HuggingFace warnings for clean output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
from src.data.dataset import VolumeReportDataset
from src.models.aligner import Multimodal3DAligner
from src.eval.retrieval import ZeroShotRetrievalEngine


def run_demo(query_text=None, checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  PS-007 MULTIMODAL 3D SELF-SUPERVISED ZERO-SHOT PATHOLOGY RETRIEVAL")
    print("=" * 70)

    # 1. Load dataset
    print("\n[Step 1] Loading 3D Scans and Clinical Reports...")
    ds = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
    print(f"Loaded {len(ds)} rare pathology volumetric cases:")
    for i in range(len(ds)):
        item = ds[i]
        print(f"  [{i+1}] {item['case_id']}: {item['pathology']} (Dimensions: {list(item['volume'].shape)})")

    # 2. Initialize Model
    print("\n[Step 2] Initializing 3D-MAE + Clinical Text Multimodal Aligner...")
    model = Multimodal3DAligner(
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        embed_dim=128,
        shared_dim=128,
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        mask_ratio=0.75
    ).to(device)

    # Load weights if checkpoint exists
    if checkpoint_path is None:
        default_ckpt = os.path.join("artifacts", "checkpoints", "multimodal_aligner.pt")
        if os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Note: No pre-existing checkpoint specified, running with freshly initialized model.")

    # 3. Index Gallery
    print("\n[Step 3] Indexing 3D scans into shared visual-textual embedding space...")
    engine = ZeroShotRetrievalEngine(model, device=device)
    engine.index_gallery(ds)
    print(f"Indexed {len(engine.gallery)} scans. Embedding vector shape: {engine.gallery_embeddings.shape[1]}-D.")

    # Inspect sample 0 representation
    sample_0 = ds[0]
    sample_0_vol = sample_0["volume"].unsqueeze(0).to(device)
    with torch.no_grad():
        sample_0_emb = model.get_image_embedding(sample_0_vol).cpu().numpy()[0]
    print(f"Sample 3D Embedding (first 8 dimensions): {np.round(sample_0_emb[:8], 4)}... (L2-norm: {np.linalg.norm(sample_0_emb):.2f})")

    # 4. Zero-Shot Retrieval Queries
    demo_queries = [
        "Thin-walled pulmonary cysts diffusely distributed throughout both lungs with preservation of lung volumes.",
        "Crazy-paving attenuation pattern caused by alveolar lipoproteinaceous filling and interlobular thickening.",
        "Large heterogeneous rim-enhancing necrotic intra-axial mass in right frontal lobe.",
        "Subpleural basilar reticular honeycombing and traction bronchiectasis.",
        "Cortical ribboning and striking bilateral basal ganglia diffusion restriction."
    ]

    queries_to_run = [query_text] if query_text else demo_queries

    print("\n[Step 4] Executing Zero-Shot Rare-Pathology Text Queries...")
    for idx, q in enumerate(queries_to_run, start=1):
        print("\n" + "-" * 70)
        print(f"QUERY {idx}: \"{q}\"")
        print("-" * 70)
        results = engine.query(q, top_k=3)
        for res in results:
            print(f"  Rank #{res['rank']} | Score: {res['similarity_score']:+.4f} | Case: {res['case_id']} | Pathology: {res['pathology']}")
            print(f"          Report: {res['clinical_report'][:75]}...")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE: Zero-shot retrieval demonstrated across rare 3D pathologies.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS-007 Zero-Shot Pathology Retrieval Demo")
    parser.add_argument("--query", type=str, default=None, help="Custom clinical text query")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    run_demo(query_text=args.query, checkpoint_path=args.checkpoint)
