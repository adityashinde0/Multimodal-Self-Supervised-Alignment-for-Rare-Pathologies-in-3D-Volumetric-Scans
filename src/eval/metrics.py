import numpy as np


def compute_average_precision(ranked_labels, true_label):
    """
    Computes Average Precision (AP) for a binary relevance ranking.
    ranked_labels: list or array of retrieved labels in rank order
    true_label: ground-truth target label
    """
    hits = 0
    sum_precisions = 0.0
    total_relevant = sum(1 for label in ranked_labels if label == true_label)
    
    if total_relevant == 0:
        return 0.0

    for rank, label in enumerate(ranked_labels, start=1):
        if label == true_label:
            hits += 1
            precision_at_k = hits / rank
            sum_precisions += precision_at_k

    return sum_precisions / total_relevant


def compute_retrieval_metrics(similarity_matrix, query_labels, gallery_labels, k_list=(1, 5, 10)):
    """
    similarity_matrix: (num_queries, num_gallery) cosine similarity scores
    query_labels: (num_queries,) ground-truth label for each query
    gallery_labels: (num_gallery,) label for each gallery item
    k_list: tuple of K values for Recall@K
    returns: dict containing mAP, Recall@K, and per-query APs
    """
    num_queries = len(query_labels)
    aps = []
    recalls = {k: 0 for k in k_list}

    for i in range(num_queries):
        true_label = query_labels[i]
        sims = similarity_matrix[i]
        
        # Sort gallery items by similarity descending
        ranked_indices = np.argsort(-sims)
        ranked_labels = [gallery_labels[idx] for idx in ranked_indices]

        # Compute AP
        ap = compute_average_precision(ranked_labels, true_label)
        aps.append(ap)

        # Compute Recall@K
        for k in k_list:
            top_k_labels = ranked_labels[:k]
            if true_label in top_k_labels:
                recalls[k] += 1

    mean_ap = float(np.mean(aps))
    recall_at_k = {f"Recall@{k}": recalls[k] / max(1, num_queries) for k in k_list}

    return {
        "mAP": mean_ap,
        **recall_at_k,
        "per_query_ap": aps
    }


def compute_relative_improvement(proposed_map, baseline_map):
    """
    Relative improvement = (proposed - baseline) / baseline
    Returns relative percentage (e.g. 0.20 for +20%)
    """
    if baseline_map == 0.0:
        return float('inf') if proposed_map > 0 else 0.0
    return (proposed_map - baseline_map) / baseline_map
