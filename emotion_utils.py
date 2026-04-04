from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL, device: str | None = None):
    device = device or choose_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=choose_dtype(device),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


@torch.no_grad()
def last_token_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    last_indices = attention_mask.sum(dim=1) - 1
    states = []
    for hidden in outputs.hidden_states[1:]:
        picked = hidden[torch.arange(hidden.size(0), device=device), last_indices]
        states.append(picked.detach().float().cpu())
    return torch.stack(states, dim=0)


def projection_scores(states: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
    return (states * vector[:, None, :]).sum(dim=-1)


def save_tensor_dict(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def load_vectors(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def best_layer_index(summary: dict[str, Any]) -> int:
    layers = summary["layers"]
    best = max(layers, key=lambda item: item["separation"])
    return int(best["layer"])


def score_single_token_options(
    model,
    tokenizer,
    prompt: str,
    option_texts: list[str],
    device: str,
) -> list[float]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    next_token_logits = logits[0, -1]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    scores = []
    for option in option_texts:
        token_ids = tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Option {option!r} is not a single token with this tokenizer: {token_ids}"
            )
        scores.append(float(log_probs[token_ids[0]].item()))
    return scores


def register_last_token_steering_hook(model, layer_index: int, vector: torch.Tensor, alpha: float):
    layers = getattr(getattr(model, "model", model), "layers")
    layer_module = layers[layer_index]
    steering = vector.detach()

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering.to(hidden.device, hidden.dtype)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering.to(hidden.device, hidden.dtype)
        return hidden

    return layer_module.register_forward_hook(hook)


def to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()
