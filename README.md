# Under Pressure: Emotional Framing & Internal Geometry in Small LLMs

Code to reproduce the findings from:

> **"Under Pressure: Emotional Framing Induces Measurable Behavioral Shifts and Structured Internal Geometry in Small Language Models"**

All experiments run on consumer hardware, no cloud or proprietary APIs needed.

---

## Core Finding

We gave Qwen 3.5 (0.8B) an impossible coding task, then varied only the emotional tone of a follow-up message across 8 conditions. Same model, same task, different words:

| Condition | Honest Rate | Hack Rate |
|-----------|-------------|-----------|
| Calm | 35% | 0% |
| Curiosity | 30% | 0% |
| Encouragement | 20% | 0% |
| Shame | 10% | 0% |
| Approval | 0% | 0% |
| Threat | 10% | 10% |
| Urgency | 15% | 15% |
| **Pressure** | **0%** | **55%** |

Internally, each condition leaves a distinct direction vector in the model's hidden states. PCA on those vectors reveals a valence axis explaining 59.5% of variance — structure the model learned from human text with no explicit supervision.

---

## Reproduce

**Requirements:** Python 3.10+, [Ollama](https://ollama.com), HuggingFace token (Qwen3.5 is gated)

```bash
git clone https://github.com/ranausmanai/LLMEmotionGeometry
cd LLMEmotionGeometry
pip install -e .
export HF_TOKEN=hf_...
```

**Step 1 — Run the behavioral benchmark**
```bash
ollama pull qwen3.5:0.8b
python run_benchmark_emotions.py    # 8 conditions × 20 runs
python analyze_benchmark_emotions.py
```

**Step 2 — Extract activation vectors**
```bash
python build_emotion_vectors.py     # downloads Qwen3.5-0.8B from HF
python build_emotion_map.py         # PCA + cosine similarity map
```

**Step 3 — Causal steering**
```bash
python steer_benchmark_choices.py
```

**Step 4 — Visualize**
```bash
python make_emotion_map_viz.py      # → results/emotion_map.html
```

Results land in `results/`. The pre-computed outputs from our runs are already there if you just want to explore.

---

## Files

| File | What it does |
|------|-------------|
| `emotion_utils.py` | Model loading, hidden state extraction, steering hooks |
| `run_benchmark_emotions.py` | 8-condition behavioral benchmark via Ollama |
| `analyze_benchmark_emotions.py` | Compute hack/honest rates per condition |
| `build_emotion_vectors.py` | Extract per-condition direction vectors from hidden states |
| `build_emotion_map.py` | PCA, k-means, cosine similarity, valence alignment |
| `steer_benchmark_choices.py` | Causal steering: inject vectors, measure behavior shift |
| `make_emotion_map_viz.py` | Interactive HTML visualization |
| `paper/` | LaTeX source + compiled PDF + figures |

---

## Paper

`paper/arxiv_paper.pdf` — full paper with methodology, results, and discussion.

To recompile: `cd paper && tectonic arxiv_paper.tex`
