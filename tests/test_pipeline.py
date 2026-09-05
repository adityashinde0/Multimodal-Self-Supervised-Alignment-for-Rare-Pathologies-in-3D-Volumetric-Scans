import unittest
import torch
import numpy as np
from src.data.dataset import VolumeReportDataset
from src.models.patch_embed import PatchEmbed3D, RandomMasking3D
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
        self.device = "cpu"

    def test_01_dataset_loading(self):
        ds = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        self.assertEqual(len(ds), 5)
        sample = ds[0]
        self.assertEqual(sample["volume"].shape, (1, 16, 16, 16))
        self.assertIn("pathology", sample)
        self.assertIn("report", sample)
        self.assertFalse(torch.isnan(sample["volume"]).any())

    def test_02_patch_embed_and_masking(self):
        patch_embed = PatchEmbed3D(img_size=(16, 16, 16), patch_size=(4, 4, 4), in_chans=1, embed_dim=128)
        self.assertEqual(patch_embed.num_patches, 64)
        self.assertEqual(patch_embed.patch_volume, 64)

        dummy_x = torch.randn(2, 1, 16, 16, 16)
        tokens = patch_embed(dummy_x)
        self.assertEqual(tokens.shape, (2, 64, 128))

        masker = RandomMasking3D(mask_ratio=0.75)
        x_vis, mask, ids_restore, ids_keep = masker(tokens)
        self.assertEqual(x_vis.shape, (2, 16, 128))  # 16 visible tokens
        self.assertEqual(mask.shape, (2, 64))
        # Mask ratio verify: 16 visible (mask==0), 48 masked (mask==1)
        self.assertEqual(int((mask == 1).sum().item()), 48 * 2)
        self.assertEqual(int((mask == 0).sum().item()), 16 * 2)

    def test_03_mae3d_forward_and_loss(self):
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

    def test_04_text_encoder_and_projection(self):
        text_enc = LightweightClinicalTextEncoder(vocab_size=1000, embed_dim=64, depth=1)
        reports = ["Thin-walled pulmonary cysts diffusely distributed.", "Necrotic mass in right frontal lobe."]
        emb = text_enc.forward_text(reports, "cpu")
        self.assertEqual(emb.shape, (2, 64))

        proj = ProjectionHead(in_dim=64, hidden_dim=128, out_dim=64)
        norm_emb = proj(emb)
        self.assertEqual(norm_emb.shape, (2, 64))
        # Check unit norm
        norms = torch.norm(norm_emb, p=2, dim=-1)
        np.testing.assert_allclose(norms.detach().numpy(), np.ones(2), atol=1e-5)

    def test_05_multimodal_aligner_and_infonce(self):
        aligner = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=64,
            shared_dim=64,
            mask_ratio=0.75,
            recon_weight=1.0
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

    def test_06_supervised_baseline(self):
        baseline = Supervised3DBaseline(in_chans=1, num_classes=5, embed_dim=64)
        dummy_vols = torch.randn(4, 1, 16, 16, 16)
        logits, emb = baseline(dummy_vols)
        self.assertEqual(logits.shape, (4, 5))
        self.assertEqual(emb.shape, (4, 64))

    def test_07_retrieval_and_metrics(self):
        # Create a mock similarity matrix
        # Query 0 matches gallery 0, Query 1 matches gallery 1
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

    def test_08_profiler(self):
        baseline = Supervised3DBaseline(in_chans=1, num_classes=5, embed_dim=64)
        profile = profile_inference_model(baseline, input_shape=(1, 1, 16, 16, 16), num_warmup=1, num_runs=3)
        self.assertTrue(profile["under_24gb_limit"])
        self.assertGreater(profile["total_parameters"], 0)
        self.assertGreater(profile["latency_ms_mean"], 0)


if __name__ == "__main__":
    unittest.main()
