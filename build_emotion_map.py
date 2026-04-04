from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_vectors(path):
    return torch.load(path, map_location="cpu")


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def pca_2d(X: torch.Tensor):
    """X: [n_points, dim] — returns coords [n_points, 2], explained_variance [2]"""
    X_centered = X - X.mean(dim=0)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ Vh[:2].T
    explained = (S ** 2 / (S ** 2).sum())[:2]
    return coords, explained.tolist(), Vh[:2]


def cosine_sim_matrix(vecs: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    keys = list(vecs.keys())
    norms = {k: vecs[k] / (vecs[k].norm() + 1e-8) for k in keys}
    matrix = {}
    for a in keys:
        matrix[a] = {}
        for b in keys:
            matrix[a][b] = round(float((norms[a] * norms[b]).sum().item()), 4)
    return matrix


def kmeans_torch(X: torch.Tensor, k: int, n_iters: int = 20, n_trials: int = 5):
    """Simple Lloyd's k-means. Returns assignments for most stable trial."""
    n = X.shape[0]
    best_assignments = None
    best_inertia = float("inf")
    for _ in range(n_trials):
        idx = torch.randperm(n)[:k]
        centroids = X[idx].clone()
        assignments = torch.zeros(n, dtype=torch.long)
        for _ in range(n_iters):
            dists = torch.cdist(X.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
            assignments = dists.argmin(dim=1)
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    centroids[j] = X[mask].mean(dim=0)
        inertia = sum(float(((X[assignments == j] - centroids[j]) ** 2).sum()) for j in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_assignments = assignments.tolist()
    return best_assignments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default="results/emotion_vectors.pt")
    parser.add_argument("--output", default="results/emotion_map.json")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to use (default: majority vote best layer)")
    return parser.parse_args()


def main():
    args = parse_args()
    vectors = load_vectors(args.vectors)

    non_baseline = vectors["non_baseline_conditions"]
    best_layers = vectors["best_layer_per_condition"]

    # Pick layer: majority vote among best layers
    if args.layer is not None:
        layer = args.layer
    else:
        layer_votes = list(best_layers.values())
        layer = max(set(layer_votes), key=layer_votes.count)
    print(f"Using layer {layer} for emotion map")

    # Stack unit vectors at chosen layer: [n_conditions, hidden]
    unit_vecs = {c: vectors["unit_vectors"][c][layer] for c in non_baseline}
    keys = list(unit_vecs.keys())
    X = torch.stack([unit_vecs[c] for c in keys])  # [n, hidden]

    # PCA
    coords_2d, explained, pc_axes = pca_2d(X)
    pca_coords = {c: [round(float(coords_2d[i, 0]), 4), round(float(coords_2d[i, 1]), 4)]
                  for i, c in enumerate(keys)}
    # calm is the origin (all vectors are deltas from calm)
    pca_coords["calm"] = [0.0, 0.0]

    print(f"PC1 explains {explained[0]*100:.1f}% | PC2 explains {explained[1]*100:.1f}%")
    print(f"Combined: {sum(explained)*100:.1f}%")

    # Cosine similarity matrix
    cosine_mat = cosine_sim_matrix(unit_vecs)

    # Add calm to cosine matrix (calm is zero vector — all sims = 0)
    all_conditions = non_baseline + ["calm"]
    for c in all_conditions:
        if c not in cosine_mat:
            cosine_mat[c] = {}
        for d in all_conditions:
            if d not in cosine_mat[c]:
                cosine_mat[c][d] = 0.0

    # K-means k=2 and k=3
    labels_k2 = kmeans_torch(X, k=2)
    labels_k3 = kmeans_torch(X, k=3)
    kmeans_2 = {c: labels_k2[i] for i, c in enumerate(keys)}
    kmeans_3 = {c: labels_k3[i] for i, c in enumerate(keys)}

    # Most and least similar pairs (excluding self)
    pairs = [(a, b, cosine_mat.get(a, {}).get(b, 0.0))
             for a in keys for b in keys if a < b]
    pairs.sort(key=lambda x: x[2], reverse=True)
    most_similar = {"a": pairs[0][0], "b": pairs[0][1], "sim": pairs[0][2]}
    least_similar = {"a": pairs[-1][0], "b": pairs[-1][1], "sim": pairs[-1][2]}

    # Valence check: project onto positive-negative axis
    positive = ["curiosity", "encouragement"]
    negative = ["shame", "threat", "pressure"]
    pos_vecs = [unit_vecs[c] for c in positive if c in unit_vecs]
    neg_vecs = [unit_vecs[c] for c in negative if c in unit_vecs]
    if pos_vecs and neg_vecs:
        valence_axis = torch.stack(neg_vecs).mean(0) - torch.stack(pos_vecs).mean(0)
        valence_axis = valence_axis / (valence_axis.norm() + 1e-8)
        pc1 = pc_axes[0] / (pc_axes[0].norm() + 1e-8)
        valence_pc1_alignment = round(float((valence_axis * pc1).sum().abs().item()), 4)
        print(f"Valence axis alignment with PC1: {valence_pc1_alignment:.3f} "
              f"({'valence IS dominant' if valence_pc1_alignment > 0.7 else 'mixed structure'})")
    else:
        valence_pc1_alignment = None

    # Separation scores at chosen layer
    sep_scores = {}
    for c in non_baseline:
        vec = vectors["layer_vectors"][c][layer]
        sep_scores[c] = round(float(vec.norm().item()), 4)
    sep_scores["calm"] = 0.0

    result = {
        "model": vectors["model"],
        "layer": layer,
        "conditions": all_conditions,
        "pca_coords": pca_coords,
        "explained_variance": [round(float(e), 4) for e in explained],
        "cosine_similarity_matrix": cosine_mat,
        "kmeans_k2": kmeans_2,
        "kmeans_k3": kmeans_3,
        "most_similar_pair": most_similar,
        "least_similar_pair": least_similar,
        "valence_pc1_alignment": valence_pc1_alignment,
        "separation_scores": sep_scores,
    }

    save_json(args.output, result)
    print(f"\nSaved {args.output}")
    print(f"\nMost similar pair: {most_similar['a']} <-> {most_similar['b']} ({most_similar['sim']:.3f})")
    print(f"Least similar pair: {least_similar['a']} <-> {least_similar['b']} ({least_similar['sim']:.3f})")
    print("\nK-means k=2 clusters:")
    for c, label in sorted(kmeans_2.items(), key=lambda x: x[1]):
        print(f"  cluster {label}: {c}")


if __name__ == "__main__":
    main()
