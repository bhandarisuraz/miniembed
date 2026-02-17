---
title: MiniEmbed Product Matcher
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# MiniEmbed: Tiny, Powerful Embedding Models from Scratch

**MiniEmbed** is a research-grade toolkit for training and deploying ultra-compact text embedding models (Bi-Encoders) built entirely from scratch in PyTorch. While the industry chases billion-parameter giants, MiniEmbed proves that a **~42 MB / 10.8M parameter** model can deliver production-grade semantic intelligence for specialized domains.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

---

## What Can MiniEmbed Do?

| Capability | Description |
|---|---|
| **Semantic Search** | Find meaning, not just keywords. Understands that *"kitten"* is similar to *"cat"*. |
| **Re-Ranking** | Sort candidates by true semantic relevance. Eliminates false positives. |
| **Clustering** | Group thousands of texts into logical categories automatically. |
| **Product Matching** | Match identical items across stores, even with messy or inconsistent titles. |
| **Text Encoding** | Convert any text into a dense 256-dimensional vector for downstream tasks. |

---

## Project Structure

```
miniembed/
|-- README.md               # You are here
|-- LICENSE                  # MIT License
|-- requirements.txt         # Python dependencies
|-- demo.py                  # Interactive Streamlit demo
|-- src/                     # Core library
|   |-- __init__.py
|   |-- model.py             # Transformer architecture (from scratch)
|   |-- tokenizer.py         # Custom word-level tokenizer
|   |-- inference.py         # High-level API for encoding & search
|-- models/
|   |-- mini/                # Pre-trained Mini model
|       |-- config.json      # Architecture blueprint
|       |-- tokenizer.json   # 30K vocabulary
|       |-- training_info.json  # Training metadata
|-- examples/                # Ready-to-run scripts
|   |-- basic_usage.py       # Encoding & similarity
|   |-- semantic_search.py   # Document retrieval
|   |-- clustering.py        # Text clustering with K-Means
|-- tests/                   # Test suite
|   |-- test_model.py        # Unit tests for all components
|-- data/
    |-- sample_data.jsonl    # 10-pair demo dataset
```

> **Note:** Model weights (`model.pt`, ~42 MB) are not included in this repository due to size constraints. See [Downloading the Pre-trained Model](#downloading-the-pre-trained-model) below.

---

## Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/bhandarisuraz/miniembed.git
cd miniembed
pip install -r requirements.txt
```

### 2. Downloading the Pre-trained Model
Download the pre-trained `model.pt` and place it in `models/mini/`:
```bash
# Option A: Download from Hugging Face (coming soon)
# Option B: Train your own (see Training section below)
```

### 3. Try It Instantly
```python
from src.inference import EmbeddingInference

model = EmbeddingInference.from_pretrained("models/mini")

# Similarity
score = model.similarity("Machine learning is great", "AI is wonderful")
print(f"Similarity: {score:.4f}")  # High score

# Semantic Search
docs = ["Python is great for AI", "I love pizza", "Neural networks learn patterns"]
results = model.search("deep learning frameworks", docs, top_k=2)
for r in results:
    print(f"  [{r['score']:.3f}] {r['text']}")
```

---

## Interactive Demo (`demo.py`)

A full-featured Streamlit dashboard for exploring the model's capabilities without writing code:

- **Similarity** -- Real-time cosine similarity between any two texts.
- **Semantic Search** -- Rank a custom document set against your query.
- **Clustering** -- Automatically categorize items using K-Means.
- **Text Encoding** -- Inspect raw 256-D vectors and their statistics.
- **CSV Matcher** -- Match records between two CSV files for deduplication or cross-platform product mapping.

```bash
streamlit run demo.py
```

---

## Testing (`tests/test_model.py`)

The test suite validates every layer of the system:

| Test Category | What It Checks |
|---|---|
| **Tokenizer** | Vocabulary integrity, padding/truncation, special token handling (`[CLS]`, `[SEP]`, `[PAD]`), unknown word mapping |
| **Architecture** | Output dimensions, gradient flow, numerical stability, embedding collapse detection |
| **L2 Normalization** | Ensures all output embeddings lie on the unit hypersphere (norm = 1.0) |
| **End-to-End** | Full pipeline from raw string to tokenization to encoding to normalized embedding |

```bash
pytest tests/test_model.py -v
```

---

## Architecture

MiniEmbed uses a **custom 4-layer Transformer encoder** built from scratch -- no HuggingFace, no pre-trained weights:

| Component | Specification |
|---|---|
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Feed-Forward Dimension | 1,024 |
| Vocabulary Size | 30,000 |
| Max Sequence Length | 128 tokens |
| Total Parameters | ~10.8M |
| Model Size on Disk | ~42 MB |
| Pooling Strategy | Mean Pooling + L2 Normalization |

### Training Objective

Training uses **Multiple Negatives Ranking Loss (MNRL)**, the industry-standard contrastive objective for Bi-Encoders:

$$\mathcal{L} = -\sum_{i=1}^{n} \log \frac{e^{sim(q_i, p_i) / \tau}}{\sum_{j=1}^{n} e^{sim(q_i, p_j) / \tau}}$$

All embeddings are **L2-normalized**, projecting text onto a unit hypersphere where cosine similarity equals dot product -- enabling ultra-fast nearest-neighbor search.

---

## Training Data Sources

The pre-trained model was trained on ~3.8 million text pairs from the following open-source datasets:

| Dataset | Type | Source |
|---|---|---|
| **Natural Questions (NQ)** | Q&A / General | [HuggingFace](https://huggingface.co/datasets/google-research-datasets/natural_questions) |
| **GooAQ** | Knowledge Search | [HuggingFace](https://huggingface.co/datasets/sentence-transformers/gooaq) |
| **WDC Product Matching** | E-commerce | [HuggingFace](https://huggingface.co/datasets/wdc/products-2017) |
| **ECInstruct** | E-commerce Tasks | [HuggingFace](https://huggingface.co/datasets/NingLab/ECInstruct) |
| **MS MARCO** | Web Search | [HuggingFace](https://huggingface.co/datasets/microsoft/ms_marco) |

> **Legal Disclaimer**: These public datasets belong to their respective stakeholders and creators. Any copyright, licensing, or legal usage constraints must be consulted with the original authors individually.

---

## Performance

Results from the pre-trained Mini model (trained on Mac M4 Pro, ~49 hours):

| Metric | Value |
|---|---|
| **Training Loss** | 0.0748 (final) |
| **Training Samples** | 3,817,707 pairs |
| **Throughput** | ~1,000 samples/sec |
| **Encoding Latency** | ~3-5 ms per text |
| **Training Epochs** | 10 |

---

## Examples

Ready-to-run scripts in the `examples/` folder:

```bash
cd examples

# Basic encoding and similarity
python basic_usage.py

# Document retrieval
python semantic_search.py

# Text clustering with K-Means
python clustering.py
```

---

## Roadmap

- **mini-product** -- A further fine-tuned version of the Mini model, specialized for high-accuracy **product matching** is Coming soon...

---

## Citation

If you use MiniEmbed in your research, please cite:

```bibtex
@software{Bhandari_MiniEmbed_2026,
  author  = {Bhandari, Suraj},
  title   = {{MiniEmbed: Tiny, Powerful Embedding Models from Scratch}},
  url     = {https://github.com/bhandarisuraz/miniembed},
  version = {1.0.0},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Explore, learn, and build smaller, smarter AI.


//self Notes

python train.py --data data/product_matching_categorized_master.jsonl --name product --size mini --max-samples 1500000 --max-seq-len 128 --epochs 10 --batch-size 256 --learning-rate 3e-5 --warmup-steps 5000 --eval-steps 1000