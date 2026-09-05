import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .data.dataset import VolumeReportDataset, get_augmented_dataset
from .models.baseline import Supervised3DBaseline
from .models.mae3d import MaskedAutoencoder3D
from .models.aligner import Multimodal3DAligner


def train_supervised_baseline(
    data_dir=".", 
    report_file="radiology_reports.json",
    num_classes=5,
    epochs=50,
    lr=1e-3,
    batch_size=8,
    device="cpu",
    seed=42
):
    """
    Trains the supervised 3D CNN baseline on pathology labels.
    Uses augmented volume dataset to prevent overfitting on 5 cases.
    """
    torch.manual_seed(seed)
    augmented_samples = get_augmented_dataset(
        data_dir=data_dir, 
        report_file=report_file, 
        num_augmentations_per_sample=15
    )

    model = Supervised3DBaseline(in_chans=1, num_classes=num_classes, embed_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        # Mini-batch shuffle
        perm = torch.randperm(len(augmented_samples))
        total_loss = 0.0
        
        for start_idx in range(0, len(augmented_samples), batch_size):
            batch_indices = perm[start_idx:start_idx + batch_size]
            batch = [augmented_samples[i] for i in batch_indices]
            
            vols = torch.stack([b["volume"] for b in batch]).to(device)
            labels = torch.stack([b["label"] for b in batch]).to(device)

            optimizer.zero_grad()
            logits, _ = model(vols)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model


def train_3d_mae_pretraining(
    data_dir=".",
    report_file="radiology_reports.json",
    epochs=50,
    lr=1e-3,
    batch_size=8,
    mask_ratio=0.75,
    device="cpu",
    seed=42
):
    """
    Pretrains 3D-MAE independently on volumetric masked reconstruction objective.
    """
    torch.manual_seed(seed)
    augmented_samples = get_augmented_dataset(
        data_dir=data_dir,
        report_file=report_file,
        num_augmentations_per_sample=15
    )

    model = MaskedAutoencoder3D(
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        embed_dim=128,
        depth=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        mask_ratio=mask_ratio
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(augmented_samples))
        for start_idx in range(0, len(augmented_samples), batch_size):
            batch_indices = perm[start_idx:start_idx + batch_size]
            batch = [augmented_samples[i] for i in batch_indices]
            vols = torch.stack([b["volume"] for b in batch]).to(device)

            optimizer.zero_grad()
            out = model(vols)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_multimodal_aligner(
    data_dir=".",
    report_file="radiology_reports.json",
    epochs=100,
    lr=1e-3,
    mask_ratio=0.5,
    recon_weight=0.2,
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    pretrained_mae_path=None,
    device="cpu",
    seed=42
):
    """
    Trains the proposed Multimodal 3D Vision-Language Aligner
    using joint 3D-MAE reconstruction + symmetric InfoNCE contrastive alignment.
    Uses distinct-case batches to prevent negative pair collisions.
    """
    torch.manual_seed(seed)
    base_ds = VolumeReportDataset(data_dir=data_dir, report_file=report_file, augment=False)
    aug_ds = VolumeReportDataset(data_dir=data_dir, report_file=report_file, augment=True)
    num_classes = len(base_ds)

    model = Multimodal3DAligner(
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        embed_dim=128,
        shared_dim=128,
        text_model_name=text_model_name,
        mask_ratio=mask_ratio,
        recon_weight=recon_weight
    ).to(device)

    # Initialize visual encoder with pretrained 3D-MAE weights if available
    if pretrained_mae_path and os.path.exists(pretrained_mae_path):
        try:
            model.visual_encoder.load_state_dict(torch.load(pretrained_mae_path, map_location=device), strict=False)
            print(f"Initialized visual encoder from pretrained MAE: {pretrained_mae_path}")
        except Exception as e:
            print(f"Note on MAE weight load: {e}")

    # Trainable parameters: visual branch, projectors, logit_scale
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    model.train()

    for epoch in range(epochs):
        # Build batch containing exactly one augmented instance of each distinct case
        batch_vols = []
        batch_reports = []
        for c in range(num_classes):
            sample = aug_ds[c]
            batch_vols.append(sample["volume"])
            batch_reports.append(sample["report"])

        vols = torch.stack(batch_vols).to(device)
        optimizer.zero_grad()
        out = model(vols, batch_reports)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model
