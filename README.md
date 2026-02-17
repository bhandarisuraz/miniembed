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
**E-commerce Product Matching & Entity Resolution**

This model is fine-tuned to solve the "Same Product, Different Description" problem in e-commerce:

*   **Marketplace Aggregation**: Unifying listings from Amazon, Walmart, and eBay into a single catalog.
*   **Competitor Analysis**: Matching your inventory against competitors to track pricing.
*   **Data Cleaning**: Removing duplicates in databases where titles vary slightly (e.g., "Nike Air Max" vs "Nike Men's Air Max Shoe").

**Example Challenges Handled:**
*   **Variations**: "iPhone 14 128GB" vs "Apple iPhone 14 Midnight 128GB"
*   **Missing Attributes**: "Sony Headphones" vs "Sony WH-1000XM5 Noise Canceling Headphones"
*   **Formatting Differences**: "5-Pack T-Shirts" vs "T-Shirt (Pack of 5)"

## Interactive Demo

This repository includes a **Streamlit** app to demonstrate the matching capability.

To run locally:

```bash
pip install -r requirements.txt
streamlit run demo.py
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