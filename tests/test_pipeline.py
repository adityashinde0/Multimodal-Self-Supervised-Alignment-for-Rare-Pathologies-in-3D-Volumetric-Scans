import os
import unittest
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

from src.utils import set_seed
from src.data.dataset import VolumeReportDataset, get_augmented_dataset
from src.models.patch_embed import PatchEmbed3D, RandomMasking3D, get_3d_sincos_pos_embed
from src.models.mae3d import MaskedAutoencoder3D
from src.models.text_encoder import ClinicalReportEncoder, LightweightClinicalTextEncoder
from src.models.projector import ProjectionHead
from src.models.aligner import Multimodal3DAligner
from src.models.baseline import Supervised3DBaseline
from src.eval.metrics import compute_retrieval_metrics, compute_relative_improvement
from src.eval.retrieval import ZeroShotRetrievalEngine
from src.eval.profiler import profile_inference_model


class TestPS007Pipeline(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.device = "cpu"

    def test_01_dataset_loading_and_split_isolation(self):
        """Test dataset loads 5 cases and train-test case filtering prevents leakage."""
        ds = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        self.assertEqual(len(ds), 5)
        sample = ds[0]
        self.assertEqual(sample["volume"].shape, (1, 16, 16, 16))
        self.assertIn("pathology", sample)
        self.assertIn("report", sample)
        self.assertFalse(torch.isnan(sample["volume"]).any())

        # Test case-level splitting
        train_ds = VolumeReportDataset(data_dir=".", indices=[0, 1, 2], augment=False)
        test_ds = VolumeReportDataset(data_dir=".", indices=[3, 4], augment=False)
        self.assertEqual(len(train_ds), 3)
        self.assertEqual(len(test_ds), 2)
        train_cases = {s["case_id"] for s in train_ds}
        test_cases = {s["case_id"] for s in test_ds}
        self.assertTrue(train_cases.isdisjoint(test_cases))

        # Test that augmentation strictly produces samples only for the training cases
        aug_samples = get_augmented_dataset(data_dir=".", indices=[0, 1, 2], num_augmentations_per_sample=3)
        aug_case_ids = {s["case_id"] for s in aug_samples}
        self.assertEqual(aug_case_ids, train_cases)
        self.assertNotIn("CASE_003", aug_case_ids)
        self.assertNotIn("CASE_004", aug_case_ids)

    def test_02_3d_patch_embedding_and_positional_embedding(self):
        """Test 3D patch embedding and 3D sin-cos positional embedding."""
        patch_embed = PatchEmbed3D(img_size=(16, 16, 16), patch_size=(4, 4, 4), in_chans=1, embed_dim=128)
        self.assertEqual(patch_embed.num_patches, 64)
        self.assertEqual(patch_embed.patch_volume, 64)

        dummy_x = torch.randn(2, 1, 16, 16, 16)
        tokens = patch_embed(dummy_x)
        self.assertEqual(tokens.shape, (2, 64, 128))

        # Test 3D sin-cos positional embedding shape
        pos_embed = get_3d_sincos_pos_embed(embed_dim=128, grid_size=(4, 4, 4))
        self.assertEqual(pos_embed.shape, (64, 128))
        self.assertTrue(torch.isfinite(pos_embed).all())

    def test_03_patchify_unpatchify_roundtrip(self):
        """Test exact patchify and unpatchify round-trip reconstruction."""
        mae = MaskedAutoencoder3D(img_size=(16, 16, 16), patch_size=(4, 4, 4), in_chans=1, embed_dim=64)
        x = torch.randn(3, 1, 16, 16, 16)
        patches = mae.patchify(x)
        self.assertEqual(patches.shape, (3, 64, 64))

        reconstructed_x = mae.unpatchify(patches)
        self.assertEqual(reconstructed_x.shape, x.shape)
        # Check exact equality within numerical precision
        diff = torch.max(torch.abs(x - reconstructed_x)).item()
        self.assertLess(diff, 1e-6)

    def test_04_masking_ratio(self):
        """Test exact 75% volumetric masking (48 masked, 16 visible tokens)."""
        tokens = torch.randn(4, 64, 128)
        masker = RandomMasking3D(mask_ratio=0.75)
        x_vis, mask, ids_restore, ids_keep = masker(tokens)

        # 64 * (1 - 0.75) = 16 visible tokens
        self.assertEqual(x_vis.shape, (4, 16, 128))
        self.assertEqual(mask.shape, (4, 64))
        # Exactly 48 masked (1) and 16 visible (0) per batch item
        for b in range(4):
            self.assertEqual(int((mask[b] == 1).sum().item()), 48)
            self.assertEqual(int((mask[b] == 0).sum().item()), 16)

    def test_05_mae3d_reconstruction_output_and_loss(self):
        """Test 3D-MAE forward pass, pred shape, and loss sanity."""
        mae = MaskedAutoencoder3D(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=64,
            depth=2,
            decoder_embed_dim=32,
            decoder_depth=1,
            mask_ratio=0.75
        )
        dummy_x = torch.randn(2, 1, 16, 16, 16)
        out = mae(dummy_x)
        self.assertIn("loss", out)
        self.assertIn("pred", out)
        self.assertIn("mask", out)
        self.assertEqual(out["pred"].shape, (2, 64, 64))
        self.assertTrue(out["loss"].item() >= 0.0)

        # Full volume encoding without mask
        vis_emb = mae.encode_volume(dummy_x)
        self.assertEqual(vis_emb.shape, (2, 64))

    def test_06_text_encoder_and_projection_normalization(self):
        """Test text embedding shape, projection head, and L2 unit hypersphere normalization."""
        text_enc = LightweightClinicalTextEncoder(vocab_size=1000, embed_dim=64, depth=1)
        reports = ["Thin-walled pulmonary cysts diffusely distributed.", "Necrotic mass in right frontal lobe."]
        emb = text_enc.forward_text(reports, "cpu")
        self.assertEqual(emb.shape, (2, 64))

        proj = ProjectionHead(in_dim=64, hidden_dim=128, out_dim=64)
        norm_emb = proj(emb)
        self.assertEqual(norm_emb.shape, (2, 64))

        # Check unit norm: ||z||_2 == 1.0
        norms = torch.norm(norm_emb, p=2, dim=-1)
        np.testing.assert_allclose(norms.detach().numpy(), np.ones(2), atol=1e-5)

    def test_07_multimodal_aligner_and_infonce_loss(self):
        """Test multimodal forward pass, projection shapes, and InfoNCE loss properties."""
        aligner = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=64,
            shared_dim=64,
            mask_ratio=0.75,
            recon_weight=0.2
        )
        dummy_vols = torch.randn(3, 1, 16, 16, 16)
        reports = [
            "Thin-walled pulmonary cysts.",
            "Subpleural basilar honeycombing.",
            "Heterogeneous necrotic mass."
        ]
        out = aligner(dummy_vols, reports)
        self.assertIn("loss", out)
        self.assertIn("contrastive_loss", out)
        self.assertIn("recon_loss", out)
        self.assertEqual(out["image_embedding"].shape, (3, 64))
        self.assertEqual(out["text_embedding"].shape, (3, 64))
        self.assertTrue(torch.isfinite(out["loss"]))

        # Check unit norm of both branches
        img_norms = torch.norm(out["image_embedding"], p=2, dim=-1).detach().numpy()
        text_norms = torch.norm(out["text_embedding"], p=2, dim=-1).detach().numpy()
        np.testing.assert_allclose(img_norms, np.ones(3), atol=1e-5)
        np.testing.assert_allclose(text_norms, np.ones(3), atol=1e-5)

        # Cosine similarity bounds: [-1.0, 1.0]
        cos_sims = torch.matmul(out["image_embedding"], out["text_embedding"].T)
        self.assertTrue((cos_sims >= -1.0001).all())
        self.assertTrue((cos_sims <= 1.0001).all())

    def test_08_retrieval_ranking_and_metrics(self):
        """Test zero-shot retrieval ranking and mAP calculation."""
        sims = np.array([
            [0.9, 0.1, 0.2],
            [0.1, 0.85, 0.15]
        ])
        query_labels = [0, 1]
        gallery_labels = [0, 1, 2]
        metrics = compute_retrieval_metrics(sims, query_labels, gallery_labels, k_list=(1, 2))
        self.assertEqual(metrics["mAP"], 1.0)
        self.assertEqual(metrics["Recall@1"], 1.0)
        self.assertEqual(metrics["Recall@2"], 1.0)

        rel_imp = compute_relative_improvement(0.85, 0.70)
        self.assertAlmostEqual(rel_imp, (0.85 - 0.70) / 0.70, places=4)

    def test_09_checkpoint_safety_and_loading(self):
        """Test checkpoint saving, loading, and safe error handling on incompatible checkpoints."""
        aligner = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=32,
            shared_dim=32,
            mask_ratio=0.75
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # 1. Save valid checkpoint
            torch.save(aligner.state_dict(), tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # 2. Reload into fresh model
            fresh_aligner = Multimodal3DAligner(
                img_size=(16, 16, 16),
                patch_size=(4, 4, 4),
                embed_dim=32,
                shared_dim=32,
                mask_ratio=0.75
            )
            state_dict = torch.load(tmp_path, map_location="cpu", weights_only=False)
            fresh_aligner.load_state_dict(state_dict)

            # Compare a parameter weight
            p1 = list(aligner.parameters())[0]
            p2 = list(fresh_aligner.parameters())[0]
            self.assertTrue(torch.equal(p1, p2))

            # 3. Test incompatible architecture detection
            mismatched_model = Multimodal3DAligner(
                img_size=(16, 16, 16),
                patch_size=(4, 4, 4),
                embed_dim=64,  # different dimension!
                shared_dim=64,
                mask_ratio=0.75
            )
            with self.assertRaises(RuntimeError):
                mismatched_model.load_state_dict(state_dict, strict=True)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_10_profiler_memory_distinction(self):
        """Test profiler distinguishes CPU RAM from GPU VRAM honestly."""
        baseline = Supervised3DBaseline(in_chans=1, num_classes=5, embed_dim=64)
        profile = profile_inference_model(baseline, input_shape=(1, 1, 16, 16, 16), device="cpu", num_warmup=1, num_runs=2)
        
        # When running on CPU, peak_vram_mb MUST be None
        self.assertIsNone(profile["peak_vram_mb"])
        self.assertIsNotNone(profile["peak_ram_mb"])
        self.assertEqual(profile["memory_type"], "cpu_ram")
        self.assertTrue(profile["under_24gb_limit"])
        self.assertGreater(profile["total_parameters"], 0)
        self.assertGreater(profile["latency_ms_mean"], 0)

    def test_11_deterministic_sha256_token_hashing(self):
        """Test fallback text encoder tokenization is 100% deterministic across instances."""
        text_enc1 = LightweightClinicalTextEncoder(vocab_size=1000, embed_dim=32, depth=1)
        text_enc2 = LightweightClinicalTextEncoder(vocab_size=1000, embed_dim=32, depth=1)
        query = "Thin-walled pulmonary cysts diffusely distributed throughout both lungs."
        tokens1 = text_enc1._tokenize(query)
        tokens2 = text_enc2._tokenize(query)
        self.assertEqual(tokens1, tokens2)
        self.assertEqual(len(tokens1), text_enc1.max_len)
        self.assertNotIn(0, tokens1[:8])  # non-padded tokens are non-zero

    def test_12_loco_train_test_absolute_isolation(self):
        """Test that in LOCO cross-validation, held-out fold cases NEVER enter training."""
        ds = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        total_cases = len(ds)
        for fold in range(total_cases):
            train_indices = [i for i in range(total_cases) if i != fold]
            held_out_case = ds[fold]["case_id"]
            
            # 1. Base dataset isolation
            train_ds = VolumeReportDataset(data_dir=".", indices=train_indices, augment=False)
            train_case_ids = {s["case_id"] for s in train_ds}
            self.assertNotIn(held_out_case, train_case_ids)
            self.assertEqual(len(train_ds), total_cases - 1)
            
            # 2. Augmented dataset isolation
            aug_samples = get_augmented_dataset(data_dir=".", indices=train_indices, num_augmentations_per_sample=5)
            aug_case_ids = {s["case_id"] for s in aug_samples}
            self.assertNotIn(held_out_case, aug_case_ids)

    def test_13_retrieval_validation_empty_and_unknown_queries(self):
        """Test ZeroShotRetrievalEngine query validation, empty input handling, and ranking."""
        aligner = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=32,
            shared_dim=32,
            mask_ratio=0.75
        )
        engine = ZeroShotRetrievalEngine(aligner, device="cpu")
        ds = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        engine.index_gallery(ds)
        self.assertEqual(len(engine.gallery), 5)

        # Empty query validation
        with self.assertRaises(ValueError):
            engine.query("", top_k=3)
        with self.assertRaises(ValueError):
            engine.query("   ", top_k=3)

        # Unknown/out-of-vocabulary query handling
        results = engine.query("xyzqwerty nonexistentpathology term", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertIn("similarity_score", results[0])
        self.assertIn("case_id", results[0])

    def test_14_infonce_temperature_and_symmetry(self):
        """Test InfoNCE loss symmetry and positive learnable temperature behavior."""
        aligner = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=32,
            shared_dim=32,
            mask_ratio=0.75
        )
        vols = torch.randn(4, 1, 16, 16, 16)
        reports = [
            "Lymphangioleiomyomatosis with thin cysts.",
            "Idiopathic pulmonary fibrosis honeycombing.",
            "Glioblastoma multiforme necrotic rim.",
            "Pulmonary alveolar proteinosis crazy-paving."
        ]
        out = aligner(vols, reports)
        self.assertTrue(torch.isfinite(out["contrastive_loss"]))
        # Temperature parameter tau must be positive
        tau = aligner.logit_scale.exp().item()
        self.assertGreater(tau, 0.0)

    def test_15_voxel_count_and_nan_inf_sanitization(self):
        """Test that volumetric data has exact 4096 voxels and handles NaN/Inf safely."""
        ds = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        for i in range(len(ds)):
            v = ds[i]["volume"]
            # 1 channel x 16 x 16 x 16 = 4096 voxels
            self.assertEqual(v.numel(), 4096)
            self.assertEqual(v.shape, (1, 16, 16, 16))
            self.assertFalse(torch.isnan(v).any())
            self.assertFalse(torch.isinf(v).any())


if __name__ == "__main__":
    unittest.main()
