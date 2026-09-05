import os
import json
import torch
import numpy as np
from src.data.dataset import VolumeReportDataset
from src.models.aligner import Multimodal3DAligner


def export_data():
    os.makedirs("frontend/js", exist_ok=True)
    os.makedirs("frontend/css", exist_ok=True)

    ds = VolumeReportDataset(data_dir=".", augment=False)
    
    # Load model to extract true 128-D embeddings
    device = "cpu"
    aligner = Multimodal3DAligner(
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        embed_dim=128,
        shared_dim=128,
        text_model_name="sentence-transformers/all-MiniLM-L6-v2"
    ).to(device)

    ckpt_path = "artifacts/checkpoints/multimodal_aligner.pt"
    if os.path.exists(ckpt_path):
        aligner.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint weights into aligner.")
    aligner.eval()

    cases_data = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            vol_tensor = item["volume"].unsqueeze(0).to(device)
            emb = aligner.get_image_embedding(vol_tensor).cpu().numpy()[0]
            
            # Extract raw 16x16x16 volume array
            vol_arr = item["volume"][0].numpy()  # (16, 16, 16)
            
            # Normalize to 0-255 uint8 for canvas rendering
            v_min, v_max = vol_arr.min(), vol_arr.max()
            norm_vol = ((vol_arr - v_min) / (v_max - v_min + 1e-6) * 255.0).astype(np.uint8)

            # Extract 2D slices at middle index (slice 8) for 3 planes
            axial_slice = norm_vol[8, :, :].tolist()      # (H, W)
            coronal_slice = norm_vol[:, 8, :].tolist()    # (D, W)
            sagittal_slice = norm_vol[:, :, 8].tolist()   # (D, H)

            # All 16 axial slices for 3D layered viewer
            axial_volume = [norm_vol[z, :, :].tolist() for z in range(16)]

            cases_data.append({
                "case_id": item["case_id"],
                "pathology": item["pathology"],
                "report": item["report"],
                "volume_file": item["volume_file"],
                "dimensions": list(item["volume"].shape),
                "embedding": [round(float(x), 4) for x in emb],
                "axial_slice": axial_slice,
                "coronal_slice": coronal_slice,
                "sagittal_slice": sagittal_slice,
                "axial_volume": axial_volume
            })

    # Read benchmark metrics
    metrics_path = "artifacts/metrics/benchmark_results.json"
    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    # Output JS file
    js_content = f"""// PS-007 Real Medical Volume Slices, Embeddings & Benchmark Data
export const PS007_CASES = {json.dumps(cases_data, indent=2)};

export const PS007_METRICS = {json.dumps(metrics_data, indent=2)};
"""
    with open("frontend/js/data.js", "w") as f:
        f.write(js_content)
    print("Successfully generated frontend/js/data.js with real scan slices and embeddings.")


if __name__ == "__main__":
    export_data()
