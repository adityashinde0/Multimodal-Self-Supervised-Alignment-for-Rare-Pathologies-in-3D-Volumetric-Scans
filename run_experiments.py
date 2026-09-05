import os
import sys
import warnings

# Suppress third-party TensorFlow/Protobuf/HuggingFace warnings for clean output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import json
import time
import torch
import numpy as np
from src.data.dataset import VolumeReportDataset
from src.train import (
    train_supervised_baseline,
    train_3d_mae_pretraining,
    train_multimodal_aligner
)
from src.eval.metrics import compute_retrieval_metrics, compute_relative_improvement
from src.eval.retrieval import ZeroShotRetrievalEngine
from src.eval.profiler import profile_inference_model


def run_all_experiments(data_dir=".", report_file="radiology_reports.json", output_dir="artifacts"):
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting PS-007 Experimental Protocol on {device.upper()} ===")

    # 1. Load ground truth test dataset (original 5 cases)
    test_ds = VolumeReportDataset(data_dir=data_dir, report_file=report_file, augment=False)
    gallery_labels = [int(test_ds[i]["label"]) for i in range(len(test_ds))]
    
    # Define clinical pathology queries (both verbatim and hallmark symptom paraphrases)
    pathology_queries = [
        # Case 0: Lymphangioleiomyomatosis
        {"query": "Thin-walled pulmonary cysts diffusely distributed throughout both lungs with preservation of lung volumes.", "label": test_ds.pathology_to_label["Lymphangioleiomyomatosis"], "pathology": "Lymphangioleiomyomatosis"},
        {"query": "Bilateral diffuse thin-walled round lung cysts preserving overall pulmonary volume.", "label": test_ds.pathology_to_label["Lymphangioleiomyomatosis"], "pathology": "Lymphangioleiomyomatosis"},
        # Case 1: Idiopathic Pulmonary Fibrosis
        {"query": "Subpleural basilar reticular opacities with architectural distortion, traction bronchiectasis, and honeycombing.", "label": test_ds.pathology_to_label["Idiopathic Pulmonary Fibrosis"], "pathology": "Idiopathic Pulmonary Fibrosis"},
        {"query": "Basilar subpleural fibrotic changes with honeycombing cystic clusters and bronchiectasis.", "label": test_ds.pathology_to_label["Idiopathic Pulmonary Fibrosis"], "pathology": "Idiopathic Pulmonary Fibrosis"},
        # Case 2: Glioblastoma Multiforme
        {"query": "Large heterogeneous rim-enhancing necrotic intra-axial mass in the right frontal lobe with extensive surrounding vasogenic edema.", "label": test_ds.pathology_to_label["Glioblastoma Multiforme"], "pathology": "Glioblastoma Multiforme"},
        {"query": "Frontal necrotic high-grade intra-axial brain neoplasm with thick irregular rim enhancement and white matter edema.", "label": test_ds.pathology_to_label["Glioblastoma Multiforme"], "pathology": "Glioblastoma Multiforme"},
        # Case 3: Pulmonary Alveolar Proteinosis
        {"query": "Bilateral symmetric ground-glass opacities with superimposed interlobular septal thickening demonstrating crazy-paving pattern.", "label": test_ds.pathology_to_label["Pulmonary Alveolar Proteinosis"], "pathology": "Pulmonary Alveolar Proteinosis"},
        {"query": "Crazy-paving attenuation pattern caused by alveolar lipoproteinaceous filling and interlobular thickening.", "label": test_ds.pathology_to_label["Pulmonary Alveolar Proteinosis"], "pathology": "Pulmonary Alveolar Proteinosis"},
        # Case 4: Creutzfeldt-Jakob Disease
        {"query": "Bilateral symmetric hyperintensity in the caudate nuclei and anterior putamina on diffusion-weighted imaging (cortical ribboning).", "label": test_ds.pathology_to_label["Creutzfeldt-Jakob Disease"], "pathology": "Creutzfeldt-Jakob Disease"},
        {"query": "Diffusion restriction ribboning in cerebral cortex and striking bilateral basal ganglia hyperintensity.", "label": test_ds.pathology_to_label["Creutzfeldt-Jakob Disease"], "pathology": "Creutzfeldt-Jakob Disease"}
    ]
    query_texts = [q["query"] for q in pathology_queries]
    query_labels = [q["label"] for q in pathology_queries]

    # ============================================================
    # 2. Experiment A: Supervised 3D Baseline
    # ============================================================
    print("\n--- Training Baseline 1: Supervised 3D CNN ---")
    baseline_model = train_supervised_baseline(
        data_dir=data_dir,
        report_file=report_file,
        epochs=60,
        device=device
    )
    torch.save(baseline_model.state_dict(), os.path.join(output_dir, "checkpoints", "baseline_supervised.pt"))
    baseline_profile = profile_inference_model(baseline_model, device=device)

    # Evaluate baseline retrieval:
    # A supervised 3D CNN predicts class posteriors P(class | volume).
    # For text retrieval, the query is mapped to candidate classes via clinical keyword matching,
    # and candidate volumes are ranked by their class posterior probabilities.
    baseline_model.eval()
    gallery_probs = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            v = test_ds[i]["volume"].unsqueeze(0).to(device)
            logits, _ = baseline_model(v)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            gallery_probs.append(probs)
    gallery_probs = np.array(gallery_probs)  # (5, num_classes)

    # Class keywords for baseline text-to-class mapping
    class_keywords = {
        0: ["lymphangioleiomyomatosis", "lam", "thin-walled", "cysts", "preservation"],
        1: ["fibrosis", "ipf", "subpleural", "honeycombing", "reticular", "bronchiectasis"],
        2: ["glioblastoma", "gbm", "rim-enhancing", "necrotic", "vasogenic", "edema", "intra-axial"],
        3: ["proteinosis", "pap", "ground-glass", "crazy-paving", "lipoproteinaceous"],
        4: ["creutzfeldt", "cjd", "caudate", "putamina", "ribboning", "diffusion"]
    }

    baseline_sims = []
    for q_item in pathology_queries:
        q_text = q_item["query"].lower()
        # Compute keyword overlap score per class
        class_scores = np.zeros(5)
        for c, kws in class_keywords.items():
            class_scores[c] = sum(1.0 for kw in kws if kw in q_text)
        if class_scores.sum() > 0:
            class_weights = class_scores / class_scores.sum()
        else:
            class_weights = np.ones(5) / 5.0
        
        # Rank volumes by expected class posterior probability
        vol_scores = np.dot(gallery_probs, class_weights)
        baseline_sims.append(vol_scores)

    baseline_sims = np.array(baseline_sims)
    baseline_metrics = compute_retrieval_metrics(baseline_sims, query_labels, gallery_labels, k_list=(1, 3, 5))

    print(f"Baseline Results -> mAP: {baseline_metrics['mAP']:.4f}, Recall@1: {baseline_metrics['Recall@1']:.4f}, Peak VRAM: {baseline_profile['peak_vram_mb']:.2f} MB, Latency: {baseline_profile['latency_ms_mean']:.2f} ms")

    # ============================================================
    # 3. Experiment B: 3D MAE Visual Pretraining (Ablation)
    # ============================================================
    print("\n--- Training Baseline 2: 3D MAE (Reconstruction Only) ---")
    mae_model = train_3d_mae_pretraining(
        data_dir=data_dir,
        report_file=report_file,
        epochs=60,
        device=device
    )
    pretrained_mae_ckpt = os.path.join(output_dir, "checkpoints", "mae3d_pretrained.pt")
    torch.save(mae_model.state_dict(), pretrained_mae_ckpt)
    mae_profile = profile_inference_model(mae_model, device=device)

    # ============================================================
    # 4. Experiment C: Proposed Model (3D-MAE + Text + InfoNCE)
    # ============================================================
    print("\n--- Training Proposed Model: 3D-MAE + Clinical Text + Symmetric Contrastive Alignment ---")
    aligner_model = train_multimodal_aligner(
        data_dir=data_dir,
        report_file=report_file,
        epochs=120,
        recon_weight=0.2,
        pretrained_mae_path=pretrained_mae_ckpt,
        device=device
    )
    torch.save(aligner_model.state_dict(), os.path.join(output_dir, "checkpoints", "multimodal_aligner.pt"))
    aligner_profile = profile_inference_model(aligner_model, device=device)

    # Evaluate zero-shot retrieval
    engine = ZeroShotRetrievalEngine(aligner_model, device=device)
    engine.index_gallery(test_ds)

    proposed_sims = []
    with torch.no_grad():
        for q_text in query_texts:
            q_emb = aligner_model.get_text_embedding([q_text]).cpu().numpy()[0]
            s = np.dot(engine.gallery_embeddings, q_emb)
            proposed_sims.append(s)
    proposed_sims = np.array(proposed_sims)
    proposed_metrics = compute_retrieval_metrics(proposed_sims, query_labels, gallery_labels, k_list=(1, 3, 5))

    # Calculate relative improvement
    rel_improvement = compute_relative_improvement(proposed_metrics["mAP"], baseline_metrics["mAP"])
    rel_percent = rel_improvement * 100.0

    print(f"\nProposed Model Results:")
    print(f"  mAP: {proposed_metrics['mAP']:.4f}")
    print(f"  Recall@1: {proposed_metrics['Recall@1']:.4f}")
    print(f"  Recall@3: {proposed_metrics['Recall@3']:.4f}")
    print(f"  Recall@5: {proposed_metrics['Recall@5']:.4f}")
    print(f"  Peak VRAM: {aligner_profile['peak_vram_mb']:.2f} MB (<= 24GB Target: {aligner_profile['under_24gb_limit']})")
    print(f"  Inference Latency: {aligner_profile['latency_ms_mean']:.2f} ms")
    print(f"  Throughput: {aligner_profile['throughput_vol_per_sec']:.1f} vol/sec")
    print(f"  Relative mAP Improvement over Baseline: {rel_percent:+.2f}% (Target: >= +15%)")

    # Compile comprehensive benchmark evidence
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "target_vram_limit_gb": 24.0
        },
        "dataset": {
            "total_cases": len(test_ds),
            "volume_shape": list(test_ds[0]["volume"].shape),
            "num_queries_evaluated": len(query_texts)
        },
        "baseline_supervised": {
            "mAP": round(baseline_metrics["mAP"], 4),
            "Recall@1": round(baseline_metrics["Recall@1"], 4),
            "Recall@3": round(baseline_metrics["Recall@3"], 4),
            "Recall@5": round(baseline_metrics["Recall@5"], 4),
            "peak_vram_mb": baseline_profile["peak_vram_mb"],
            "latency_ms": baseline_profile["latency_ms_mean"],
            "parameters": baseline_profile["total_parameters"]
        },
        "proposed_multimodal_mae": {
            "mAP": round(proposed_metrics["mAP"], 4),
            "Recall@1": round(proposed_metrics["Recall@1"], 4),
            "Recall@3": round(proposed_metrics["Recall@3"], 4),
            "Recall@5": round(proposed_metrics["Recall@5"], 4),
            "peak_vram_mb": aligner_profile["peak_vram_mb"],
            "peak_vram_gb": aligner_profile["peak_vram_gb"],
            "latency_ms": aligner_profile["latency_ms_mean"],
            "throughput_vol_per_sec": aligner_profile["throughput_vol_per_sec"],
            "parameters": aligner_profile["total_parameters"]
        },
        "comparison": {
            "baseline_mAP": round(baseline_metrics["mAP"], 4),
            "proposed_mAP": round(proposed_metrics["mAP"], 4),
            "absolute_diff": round(proposed_metrics["mAP"] - baseline_metrics["mAP"], 4),
            "relative_improvement_pct": round(rel_percent, 2),
            "target_15_pct_achieved": rel_percent >= 15.0,
            "vram_target_24gb_met": aligner_profile["under_24gb_limit"]
        }
    }

    metrics_file = os.path.join(output_dir, "metrics", "benchmark_results.json")
    with open(metrics_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nBenchmark evidence saved to {metrics_file}")
    return results_payload


if __name__ == "__main__":
    run_all_experiments()
