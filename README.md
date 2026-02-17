---
title: MiniEmbed Product Matcher
emoji: ""
colorFrom: blue
colorTo: indigo
pinned: false
license: mit
library_name: generic
tags:
- embeddings
- product-matching
---

# MiniEmbed: Product Matching Model

This is a specialized version of **MiniEmbed**, fine-tuned exclusively for **high-accuracy product matching** (entity resolution). 

Unlike general-purpose embedding models, this model is designed to determine if two product listings—often with different titles, specifications, or formatting—refer to the **exact same physical item**.

## Use Case

**Cross-Catalog Product Matching**
*   **Scenario**: You have a catalog (Site A) and want to find matching products in a competitor's catalog (Site B).
*   **Challenge**: Titles differ ("iPhone 14 128GB" vs "Apple iPhone 14 Midnight 128GB"), specs are formatted differently, and noise/distractors exist.
*   **Solution**: This model maps semantically identical products to the same vector space, ignoring irrelevant noise while paying attention to critical specs (GB, Model Number, Color).

## Interactive Demo

This repository includes a **Streamlit** app to demonstrate the matching capability.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Architecture

*   **Type**: Transformer Bi-Encoder (BERT-style)
*   **Parameters**: ~10.8M (Mini)
*   **Dimensions**: 256
*   **Max Sequence Length**: 128 tokens
*   **Format**: `SafeTensors` (Hugging Face ready)

## Usage

Since this is a custom model, you need to download the code and weights from the Hub:

```python
from huggingface_hub import snapshot_download
import sys

# 1. Download model (one-time)
model_dir = snapshot_download("surazbhandari/miniembed-product")

# 2. Add to path so we can import 'src'
sys.path.insert(0, model_dir)

# 3. Load Model
from src.inference import EmbeddingInference
model = EmbeddingInference.from_pretrained(model_dir)

# Define two product titles
product_a = "Sony WH-1000XM5 Wireless Noise Canceling Headphones, Black"
product_b = "Sony WH1000XM5/B Headphones"

# Calculate similarity (0 to 1)
score = model.similarity(product_a, product_b)

print(f"Similarity: {score:.4f}")
```

## Automated Sync

This repository is automatically synced to Hugging Face Spaces via GitHub Actions.


MIT