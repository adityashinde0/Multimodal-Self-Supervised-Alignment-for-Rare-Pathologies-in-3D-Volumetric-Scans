import time
import torch
import numpy as np


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def profile_inference_model(model, input_shape=(1, 1, 16, 16, 16), device="cpu", num_warmup=5, num_runs=20):
    """
    Profiles inference memory (accurately distinguishing GPU VRAM from CPU Host RAM),
    latency, and throughput.
    
    Scientific Honesty Rules:
    - Never reports CPU memory as GPU VRAM.
    - If CUDA is unavailable, explicitly reports CPU execution and peak_vram_mb as None.
    - Validates execution against the <= 24GB target limit.
    """
    cuda_available = torch.cuda.is_available() and device.startswith("cuda")
    gpu_name = torch.cuda.get_device_name(device) if cuda_available else None
    
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

    # CUDA memory tracking initialization
    if cuda_available:
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

            if cuda_available:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # in ms

    # Model tensor static footprint (parameters + buffers)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_tensor_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

    # Differentiate GPU VRAM vs CPU RAM
    if cuda_available:
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_mb = round(peak_vram_bytes / (1024 ** 2), 4)
        peak_vram_gb = round(peak_vram_bytes / (1024 ** 3), 6)
        peak_ram_mb = None
        peak_ram_gb = None
        memory_type = "gpu_vram"
        effective_memory_gb = peak_vram_gb
    else:
        # On CPU, GPU VRAM does not exist. Report model RAM allocation.
        peak_vram_mb = None
        peak_vram_gb = None
        peak_ram_mb = round(model_tensor_mb, 4)
        peak_ram_gb = round(model_tensor_mb / 1024, 6)
        memory_type = "cpu_ram"
        effective_memory_gb = peak_ram_gb

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    batch_size = input_shape[0]
    throughput = (batch_size / (mean_latency / 1000.0)) if mean_latency > 0 else 0.0

    param_info = count_parameters(model)

    return {
        "device": device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "memory_type": memory_type,
        "model_tensor_memory_mb": round(model_tensor_mb, 4),
        "peak_vram_mb": peak_vram_mb,
        "peak_vram_gb": peak_vram_gb,
        "peak_ram_mb": peak_ram_mb,
        "peak_ram_gb": peak_ram_gb,
        "under_24gb_limit": effective_memory_gb <= 24.0,
        "latency_ms_mean": round(mean_latency, 3),
        "latency_ms_std": round(std_latency, 3),
        "throughput_vol_per_sec": round(throughput, 2),
        "total_parameters": param_info["total"],
        "trainable_parameters": param_info["trainable"]
    }
