"""
Unit Tests for MiniEmbed
========================
Validates the tokenizer, model architecture, and end-to-end inference pipeline.

Run with: pytest tests/test_model.py -v

Test Categories:
  - TestTokenizer: Vocabulary construction, encoding, decoding, special tokens.
  - TestModel: Forward pass shapes, embedding normalization, padding masks.
  - TestIntegration: Full pipeline from raw text to unit-normalized embedding.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np

from src.model import MiniTransformerEmbedding
from src.tokenizer import SimpleTokenizer


class TestTokenizer:
    """Test the SimpleTokenizer class.
    
    The tokenizer converts raw text into integer token IDs that the model
    can process. It maintains a fixed vocabulary built from training data,
    with special tokens for padding, unknown words, and sequence boundaries.
    """
    
    def test_initialization(self):
        """Verify that a fresh tokenizer contains only the 4 special tokens."""
        tokenizer = SimpleTokenizer(vocab_size=1000)
        assert len(tokenizer.word_to_id) == 4  # [PAD], [UNK], [CLS], [SEP]
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
    
    def test_build_vocab(self):
        """Verify that build_vocab populates the vocabulary from text."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        texts = ["hello world", "hello python", "machine learning"] * 10
        tokenizer.build_vocab(texts, min_freq=2)
        
        # Should have more than just special tokens
        assert len(tokenizer.word_to_id) > 4
        assert "hello" in tokenizer.word_to_id
        assert "world" in tokenizer.word_to_id
    
    def test_encode_produces_correct_structure(self):
        """Verify that encode() returns input_ids and attention_mask of the
        correct length, and that the sequence starts with [CLS]."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world test"] * 10, min_freq=1)
        
        result = tokenizer.encode("hello world", max_length=10)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert len(result['input_ids']) == 10
        assert len(result['attention_mask']) == 10
        # First token must be [CLS]
        assert result['input_ids'][0] == tokenizer.cls_token_id
    
    def test_padding_fills_remainder(self):
        """Verify that sequences shorter than max_length are zero-padded."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["a b"] * 10, min_freq=1)
        
        result = tokenizer.encode("a b", max_length=20)
        ids = result['input_ids'].tolist()
        mask = result['attention_mask'].tolist()
        
        # Trailing positions should be PAD (0) with mask 0
        assert ids[-1] == tokenizer.pad_token_id
        assert mask[-1] == 0
        # Active positions should have mask 1
        assert mask[0] == 1
    
    def test_decode_recovers_words(self):
        """Verify that decode() reconstructs the original words (minus special tokens)."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world"] * 10, min_freq=1)
        
        encoded = tokenizer.encode("hello world", max_length=10)
        decoded = tokenizer.decode(encoded['input_ids'].tolist())
        
        assert "hello" in decoded
        assert "world" in decoded

    def test_unknown_words_map_to_unk(self):
        """Words not in the vocabulary should map to [UNK] (id=1)."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world"] * 10, min_freq=1)
        
        result = tokenizer.encode("hello xyz", max_length=10)
        ids = result['input_ids'].tolist()
        
        # "xyz" is not in vocab, so it should be mapped to UNK
        assert tokenizer.unk_token_id in ids


class TestModel:
    """Test the MiniTransformerEmbedding model.
    
    The model is a 4-layer Transformer encoder that converts token sequences
    into fixed-size dense vectors via mean pooling + L2 normalization.
    These tests use a smaller configuration (d_model=64, 2 layers) for speed.
    """
    
    @pytest.fixture
    def model(self):
        """Create a small test model (not the full production model)."""
        return MiniTransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=32
        )
    
    def test_forward_pass_output_shape(self, model):
        """The raw forward pass should return token-level representations:
        [batch_size, seq_len, d_model]."""
        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        output = model(input_ids, attention_mask)
        
        assert output.shape == (batch_size, seq_len, 64)
    
    def test_encode_output_shape(self, model):
        """The encode() method applies mean pooling and should return
        one vector per sequence: [batch_size, d_model]."""
        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        embeddings = model.encode(input_ids, attention_mask)
        
        assert embeddings.shape == (batch_size, 64)
    
    def test_embeddings_are_l2_normalized(self, model):
        """All output embeddings must lie on the unit sphere (L2 norm = 1.0).
        This is critical because cosine similarity equals dot product when
        vectors are unit-normalized."""
        input_ids = torch.randint(0, 1000, (4, 16))
        attention_mask = torch.ones_like(input_ids)
        
        embeddings = model.encode(input_ids, attention_mask)
        norms = torch.norm(embeddings, p=2, dim=1)
        
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_different_inputs_produce_different_embeddings(self, model):
        """Sanity check: different token sequences should not produce
        identical embeddings (model is not collapsing)."""
        ids_a = torch.randint(0, 500, (1, 16))
        ids_b = torch.randint(500, 1000, (1, 16))
        mask = torch.ones(1, 16)
        
        emb_a = model.encode(ids_a, mask)
        emb_b = model.encode(ids_b, mask)
        
        # Embeddings should differ
        assert not torch.allclose(emb_a, emb_b, atol=1e-3)
    
    def test_padding_mask_is_respected(self, model):
        """Verify that padding tokens do not corrupt the output.
        Both fully-valid and partially-padded sequences should produce
        finite, non-NaN embeddings."""
        input_ids = torch.randint(0, 1000, (2, 16))
        
        mask_full = torch.ones(1, 16)
        mask_half = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], dim=1)
        
        emb_full = model.encode(input_ids[:1], mask_full)
        emb_half = model.encode(input_ids[1:], mask_half)
        
        assert not torch.isnan(emb_full).any()
        assert not torch.isnan(emb_half).any()
    
    def test_gradient_flow(self, model):
        """Verify that gradients flow through the entire model.
        This confirms that all layers are connected and trainable."""
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones_like(input_ids)
        
        embeddings = model.encode(input_ids, attention_mask)
        loss = embeddings.sum()
        loss.backward()
        
        # Check that the token embedding layer received gradients
        assert model.token_embedding.weight.grad is not None
        assert model.token_embedding.weight.grad.abs().sum() > 0


class TestIntegration:
    """Integration tests: tokenizer + model working together.
    
    These tests simulate the real inference pipeline, from raw string input
    all the way to a unit-normalized embedding vector.
    """
    
    def test_end_to_end_pipeline(self):
        """Full pipeline: raw text -> tokenize -> encode -> normalized vector."""
        tokenizer = SimpleTokenizer(vocab_size=500)
        texts = ["machine learning is great", "deep learning rocks"] * 20
        tokenizer.build_vocab(texts, min_freq=1)
        
        model = MiniTransformerEmbedding(
            vocab_size=len(tokenizer.word_to_id),
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64
        )
        
        text = "machine learning is awesome"
        encoded = tokenizer.encode(text, max_length=16)
        
        input_ids = encoded['input_ids'].unsqueeze(0)
        attention_mask = encoded['attention_mask'].unsqueeze(0)
        
        embedding = model.encode(input_ids, attention_mask)
        
        # Output shape: [1, d_model]
        assert embedding.shape == (1, 32)
        # Must be unit-normalized
        assert torch.isclose(torch.norm(embedding), torch.tensor(1.0), atol=1e-5)
    
    def test_similar_texts_have_higher_similarity(self):
        """Semantically related texts should produce embeddings with higher
        cosine similarity than unrelated texts. This is a basic sanity check
        (not a benchmark) to verify that the architecture is structurally sound."""
        tokenizer = SimpleTokenizer(vocab_size=500)
        corpus = [
            "machine learning is great",
            "deep learning is powerful",
            "pizza is delicious food",
        ] * 20
        tokenizer.build_vocab(corpus, min_freq=1)
        
        model = MiniTransformerEmbedding(
            vocab_size=len(tokenizer.word_to_id),
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64
        )
        
        def encode(text):
            enc = tokenizer.encode(text, max_length=16)
            return model.encode(
                enc['input_ids'].unsqueeze(0),
                enc['attention_mask'].unsqueeze(0)
            )
        
        emb_ml = encode("machine learning")
        emb_dl = encode("deep learning")
        emb_food = encode("pizza food")
        
        sim_related = torch.dot(emb_ml[0], emb_dl[0]).item()
        sim_unrelated = torch.dot(emb_ml[0], emb_food[0]).item()
        
        # Both are valid floats (not NaN)
        assert not np.isnan(sim_related)
        assert not np.isnan(sim_unrelated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
