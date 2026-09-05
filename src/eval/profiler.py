import time
import torch
import numpy as np


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def profile_inference_model(model, input_shape=(1, 1, 16, 16, 16), device="cpu", num_warmup=5, num_runs=20):
    """
    Profiles inference VRAM (if CUDA is available) or host memory, latency, and throughput.
    input_shape: (B, C, D, H, W)
    """
    model = model.to(device)
    model.eval()

    dummy_vol = torch.randn(input_shape, device=device)
    dummy_text = ["A clinical pathology report with diffuse bilateral opacities and honeycombing."]

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if hasattr(model, "get_image_embedding"):
                _ = model.get_image_embedding(dummy_vol)
                _ = model.get_text_embedding(dummy_text)
            elif hasattr(model, "extract_features"):
                _ = model.extract_features(dummy_vol)
            else:
                _ = model(dummy_vol)

    # CUDA memory tracking
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Latency measurement
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            if hasattr(model, "get_image_embedding"):
                _ = model.get_image_embedding(dummy_vol)
            elif hasattr(model, "extract_features"):
                _ = model.extract_features(dummy_vol)
            else:
                _ = model(dummy_vol)

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # in ms

    # Peak VRAM
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    else:
        # For CPU execution, estimate model tensor allocation memory in MB
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        peak_vram_bytes = param_bytes + buffer_bytes
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    batch_size = input_shape[0]
    throughput = (batch_size / (mean_latency / 1000.0)) if mean_latency > 0 else 0.0

    param_info = count_parameters(model)

    return {
        "device": device,
        "input_shape": list(input_shape),
        "batch_size": batch_size,
        "peak_vram_mb": round(peak_vram_mb, 4),
        "peak_vram_gb": round(peak_vram_gb, 6),
        "under_24gb_limit": peak_vram_gb <= 24.0,
        "latency_ms_mean": round(mean_latency, 3),
        "latency_ms_std": round(std_latency, 3),
        "throughput_vol_per_sec": round(throughput, 2),
        "total_parameters": param_info["total"],
        "trainable_parameters": param_info["trainable"]
    }
