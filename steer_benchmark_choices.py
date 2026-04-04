from __future__ import annotations

import argparse

from emotion_utils import (
    load_json,
    load_model_and_tokenizer,
    load_vectors,
    register_last_token_steering_hook,
    save_json,
    score_single_token_options,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Score A/B benchmark-hack choices under steering.")
    parser.add_argument("--prompts", default="data/benchmark_choice_prompts.json")
    parser.add_argument("--vectors", default="results/benchmark_vectors_v2.pt")
    parser.add_argument("--output", default="results/benchmark_choice_steering.json")
    parser.add_argument("--alpha", type=float, default=4.0)
    return parser.parse_args()


def prob_b(logp_a: float, logp_b: float) -> float:
    import math

    max_logp = max(logp_a, logp_b)
    pa = math.exp(logp_a - max_logp)
    pb = math.exp(logp_b - max_logp)
    return pb / (pa + pb)


def score_condition(model, tokenizer, device, prompt: str, layer: int, vector, alpha: float):
    handle = None
    if alpha != 0.0:
        handle = register_last_token_steering_hook(model, layer, vector, alpha)
    try:
        logp_a, logp_b = score_single_token_options(model, tokenizer, prompt, [" A", " B"], device)
    finally:
        if handle is not None:
            handle.remove()
    return {
        "logp_A": logp_a,
        "logp_B": logp_b,
        "margin_B_minus_A": logp_b - logp_a,
        "prob_B": prob_b(logp_a, logp_b),
        "choice": "A" if logp_a >= logp_b else "B",
    }


def main():
    args = parse_args()
    prompts = load_json(args.prompts)
    vectors = load_vectors(args.vectors)
    model, tokenizer, device = load_model_and_tokenizer(vectors["model"])
    layer = int(vectors["best_layer"])
    vector = vectors["unit_vectors"][layer]
    items = []
    summary = {"baseline_prob_B": 0.0, "pressure_prob_B": 0.0, "calm_prob_B": 0.0}

    for item in prompts:
        baseline = score_condition(model, tokenizer, device, item["prompt"], layer, vector, 0.0)
        pressure = score_condition(model, tokenizer, device, item["prompt"], layer, vector, args.alpha)
        calm = score_condition(model, tokenizer, device, item["prompt"], layer, vector, -args.alpha)
        items.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "baseline": baseline,
                "pressure_steer": pressure,
                "calm_steer": calm,
            }
        )
        summary["baseline_prob_B"] += baseline["prob_B"]
        summary["pressure_prob_B"] += pressure["prob_B"]
        summary["calm_prob_B"] += calm["prob_B"]

    n = len(prompts)
    for key in summary:
        summary[key] /= n

    save_json(
        args.output,
        {
            "model": vectors["model"],
            "vector_type": vectors["vector_type"],
            "layer": layer,
            "alpha": args.alpha,
            "summary": summary,
            "items": items,
        },
    )
    print(f"Saved {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
