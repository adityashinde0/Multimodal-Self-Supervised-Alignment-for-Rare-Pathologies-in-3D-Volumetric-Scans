import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.dataset import VolumeReportDataset
from src.models.aligner import Multimodal3DAligner
from src.eval.retrieval import ZeroShotRetrievalEngine
from src.eval.metrics import compute_retrieval_metrics

def test_alignment():
    device = "cpu"
    base_ds = VolumeReportDataset(data_dir=".", augment=False)
    aug_ds = VolumeReportDataset(data_dir=".", augment=True)
    
    aligner = Multimodal3DAligner(
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embed_dim=128, 
        shared_dim=128, 
        mask_ratio=0.5, 
        recon_weight=0.2
    )
    
    # Load pretrained MAE weights if available
    mae_weights = "artifacts/checkpoints/mae3d_pretrained.pt"
    try:
        aligner.visual_encoder.load_state_dict(torch.load(mae_weights), strict=False)
        print("Loaded MAE pretrained weights into visual encoder.")
    except Exception as e:
        print("Pretrained MAE load note:", e)

    optimizer = optim.AdamW(aligner.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    print("Training aligner with distinct-case batches and cosine scheduler...")
    for epoch in range(200):
        # Build a batch containing exactly one augmented version of each of the 5 cases
        batch_vols = []
        batch_reports = []
        for c in range(5):
            sample = aug_ds[c]
            batch_vols.append(sample["volume"])
            batch_reports.append(sample["report"])

        vols = torch.stack(batch_vols).to(device)
        optimizer.zero_grad()
        out = aligner(vols, batch_reports)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Total Loss = {loss.item():.4f}, Contrastive = {out['contrastive_loss'].item():.4f}, Recon = {out['recon_loss'].item():.4f}")

    aligner.eval()
    engine = ZeroShotRetrievalEngine(aligner, device=device)
    engine.index_gallery(base_ds)

    print("\nEvaluating retrieval on 5 ground truth cases:")
    correct_count = 0
    for i in range(5):
        item = base_ds[i]
        res = engine.query(item["report"], top_k=5)
        is_top = (res[0]["pathology"] == item["pathology"])
        if is_top: correct_count += 1
        pathologies = [r["pathology"] for r in res]
        rank = pathologies.index(item["pathology"]) + 1
        print(f"Case {i} ({item['pathology']}): Rank #{rank} (score={res[0]['similarity_score']:.3f}) -> {'PASS' if is_top else 'FAIL'}")

    print(f"\nTop-1 Accuracy: {correct_count}/5 ({correct_count/5.0 * 100:.1f}%)")

if __name__ == "__main__":
    test_alignment()
