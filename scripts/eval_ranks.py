import torch
from src.data.dataset import VolumeReportDataset
from src.models.aligner import Multimodal3DAligner
from src.eval.retrieval import ZeroShotRetrievalEngine

def main():
    ds = VolumeReportDataset(data_dir=".", augment=False)
    model = Multimodal3DAligner(embed_dim=128, shared_dim=128)
    model.load_state_dict(torch.load("artifacts/checkpoints/multimodal_aligner.pt"))
    engine = ZeroShotRetrievalEngine(model)
    engine.index_gallery(ds)

    for i in range(len(ds)):
        item = ds[i]
        q = item["report"]
        res = engine.query(q, top_k=5)
        pathologies = [r["pathology"] for r in res]
        correct_rank = pathologies.index(item["pathology"]) + 1
        top_match = res[0]["pathology"]
        top_score = res[0]["similarity_score"]
        print(f"Case {i} ({item['pathology']}):")
        print(f"  Top Match: {top_match} (score={top_score:.3f})")
        print(f"  Correct Rank: {correct_rank}")
        for r in res:
            print(f"    #{r['rank']}: {r['pathology']} ({r['similarity_score']:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    main()
