---
title: MiniEmbed Product Matcher
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
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

You can use the provided `src` library to run inference in your own Python scripts:

```python
from src.inference import EmbeddingInference

# Load model from current directory
model = EmbeddingInference.from_pretrained(".")

# Define two product titles
product_a = "Sony WH-1000XM5 Wireless Noise Canceling Headphones, Black"
product_b = "Sony WH1000XM5/B Headphones"

# Calculate similarity (0 to 1)
score = model.similarity(product_a, product_b)

if score > 0.82:
    print(f"It's a match! (Score: {score:.4f})")
else:
    print(f"Different products. (Score: {score:.4f})")
```

## License

MIT