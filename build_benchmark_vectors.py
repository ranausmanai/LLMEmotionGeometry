from __future__ import annotations

import argparse

from emotion_utils import (
    DEFAULT_MODEL,
    last_token_hidden_states,
    load_json,
    load_model_and_tokenizer,
    save_json,
    save_tensor_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build pressure-minus-calm vectors from benchmark final replies.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default="results/benchmark_pressure_runs.json")
    parser.add_argument("--output", default="results/benchmark_vectors.pt")
    parser.add_argument("--summary", default="results/benchmark_vector_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_json(args.input)
    pair_map = {}
    for run in data["runs"]:
        key = (run["task_id"], run["seed"] % 1000)
        pair_map.setdefault(key, {})[run["condition"]] = run["turns"][-1]["assistant"]

    pressure_texts = []
    calm_texts = []
    for pair in pair_map.values():
        if "pressure" in pair and "calm" in pair:
            pressure_texts.append(pair["pressure"])
            calm_texts.append(pair["calm"])

    model, tokenizer, device = load_model_and_tokenizer(args.model)
    pressure_states = last_token_hidden_states(model, tokenizer, pressure_texts, device=device)
    calm_states = last_token_hidden_states(model, tokenizer, calm_texts, device=device)
    vector = pressure_states.mean(dim=1) - calm_states.mean(dim=1)
    unit_vector = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
    separation = ((pressure_states - calm_states) * unit_vector[:, None, :]).sum(dim=-1)

    layers = []
    for idx in range(vector.size(0)):
        layers.append(
            {
                "layer": idx,
                "separation": float(separation[idx].mean().item()),
                "vector_norm": float(vector[idx].norm().item()),
            }
        )
    best = max(layers, key=lambda x: x["separation"])
    save_tensor_dict(
        args.output,
        {
            "model": args.model,
            "vector_type": "benchmark_pressure_minus_calm_final_replies",
            "layer_vectors": vector,
            "unit_vectors": unit_vector,
            "layers": layers,
            "best_layer": best["layer"],
        },
    )
    save_json(
        args.summary,
        {
            "model": args.model,
            "vector_type": "benchmark_pressure_minus_calm_final_replies",
            "best_layer": best["layer"],
            "layers": layers,
            "pairs": len(pressure_texts),
        },
    )
    print(f"Saved {args.output}")
    print(f"Saved {args.summary}")
    print("Best layer:", best["layer"], "separation=", round(best["separation"], 4))


if __name__ == "__main__":
    main()
