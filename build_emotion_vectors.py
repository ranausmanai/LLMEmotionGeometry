from __future__ import annotations

import argparse

import torch

from emotion_utils import (
    last_token_hidden_states,
    load_json,
    load_model_and_tokenizer,
    save_json,
    save_tensor_dict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--input", default="results/benchmark_emotion_runs.json")
    parser.add_argument("--output", default="results/emotion_vectors.pt")
    parser.add_argument("--summary", default="results/emotion_vector_summary.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--baseline", default="calm")
    return parser.parse_args()


def batched_hidden_states(model, tokenizer, texts, device, batch_size):
    parts = []
    for i in range(0, len(texts), batch_size):
        parts.append(last_token_hidden_states(model, tokenizer, texts[i:i+batch_size], device=device))
    # shape: [n_layers, n_texts, hidden]
    return torch.cat(parts, dim=1)


def main():
    args = parse_args()
    data = load_json(args.input)
    conditions = data.get("conditions") or sorted({r["condition"] for r in data["runs"]})

    # group final-turn texts per condition
    texts_per_condition: dict[str, list[str]] = {c: [] for c in conditions}
    for run in data["runs"]:
        c = run["condition"]
        if c in texts_per_condition:
            texts_per_condition[c].append(run["turns"][-1]["assistant"])

    model, tokenizer, device = load_model_and_tokenizer(args.model)

    print("Extracting hidden states per condition...")
    mean_states: dict[str, torch.Tensor] = {}
    for c in conditions:
        texts = texts_per_condition[c]
        if not texts:
            continue
        states = batched_hidden_states(model, tokenizer, texts, device, args.batch_size)
        # mean over samples: [n_layers, hidden]
        mean_states[c] = states.float().mean(dim=1)
        print(f"  {c}: {len(texts)} texts, shape {mean_states[c].shape}")

    baseline = args.baseline
    non_baseline = [c for c in conditions if c != baseline]

    # direction vectors: condition - baseline
    layer_vectors: dict[str, torch.Tensor] = {}
    unit_vectors: dict[str, torch.Tensor] = {}
    for c in non_baseline:
        vec = mean_states[c] - mean_states[baseline]  # [n_layers, hidden]
        layer_vectors[c] = vec
        unit_vectors[c] = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)

    # per-condition per-layer separation score
    layers_summary: dict[str, list[dict]] = {}
    for c in non_baseline:
        uv = unit_vectors[c]
        # project mean difference onto unit vector per layer
        scores = []
        for layer_idx in range(uv.shape[0]):
            sep = float(layer_vectors[c][layer_idx].norm().item())
            scores.append({"layer": layer_idx, "separation": sep, "vector_norm": sep})
        layers_summary[c] = scores

    best_layers = {}
    for c in non_baseline:
        best = max(layers_summary[c], key=lambda x: x["separation"])
        best_layers[c] = best["layer"]
        print(f"  {c}: best_layer={best['layer']} separation={best['separation']:.4f}")

    save_tensor_dict(args.output, {
        "model": args.model,
        "baseline": baseline,
        "conditions": conditions,
        "non_baseline_conditions": non_baseline,
        "layer_vectors": layer_vectors,
        "unit_vectors": unit_vectors,
        "mean_states": mean_states,
        "best_layer_per_condition": best_layers,
    })

    save_json(args.summary, {
        "model": args.model,
        "baseline": baseline,
        "conditions": conditions,
        "non_baseline_conditions": non_baseline,
        "layers_per_condition": layers_summary,
        "best_layer_per_condition": best_layers,
    })

    print(f"\nSaved {args.output}")
    print(f"Saved {args.summary}")


if __name__ == "__main__":
    main()
