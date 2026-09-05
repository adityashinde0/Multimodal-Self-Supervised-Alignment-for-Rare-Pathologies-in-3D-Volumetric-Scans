import torch
import numpy as np


class ZeroShotRetrievalEngine:
    """
    Zero-Shot Pathology Retrieval Engine.
    Aligns free-text clinical queries with stored 3D volumetric scans.
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.gallery = []
        self.gallery_embeddings = None

    def index_gallery(self, dataset):
        """
        Extracts and stores 3D scan embeddings for all volumes in dataset.
        dataset: VolumeReportDataset
        """
        self.model.eval()
        embeddings = []
        self.gallery = []

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                volume = sample["volume"].unsqueeze(0).to(self.device)
                
                # Check if model has get_image_embedding (multimodal aligner) or extract_features (baseline)
                if hasattr(self.model, "get_image_embedding"):
                    emb = self.model.get_image_embedding(volume)
                elif hasattr(self.model, "extract_features"):
                    emb = self.model.extract_features(volume)
                else:
                    raise AttributeError("Model must have get_image_embedding or extract_features method")

                embeddings.append(emb.cpu().numpy()[0])
                self.gallery.append({
                    "case_id": sample["case_id"],
                    "pathology": sample["pathology"],
                    "report": sample["report"],
                    "label": int(sample["label"]),
                    "volume_file": sample["volume_file"]
                })

        self.gallery_embeddings = np.array(embeddings)  # (N, D)
        return len(self.gallery)

    def query(self, text_query, top_k=5):
        """
        Performs zero-shot retrieval given a clinical text description.
        returns: list of dicts ranked by cosine similarity
        """
        if self.gallery_embeddings is None or len(self.gallery) == 0:
            raise ValueError("Gallery is empty. Call index_gallery first.")

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "get_text_embedding"):
                text_emb = self.model.get_text_embedding([text_query]).cpu().numpy()[0]
            else:
                raise AttributeError("Model does not support direct text query embedding.")

        # Compute cosine similarity (embeddings are already L2-normalized)
        sims = np.dot(self.gallery_embeddings, text_emb)
        ranked_indices = np.argsort(-sims)

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            item = self.gallery[idx]
            results.append({
                "rank": rank,
                "case_id": item["case_id"],
                "pathology": item["pathology"],
                "similarity_score": float(sims[idx]),
                "clinical_report": item["report"],
                "volume_file": item["volume_file"]
            })

        return results
