"""Tests for BitNet Transformer implementation."""

import math
from typing import Dict, Any

import pytest
import torch
import numpy as np

from qraf.core.bitnet_transformer import (
    BitNetConfig,
    BitNetTokenizer,
    BitNetTransformer,
    BitNetAttention,
    BitNetDecoder,
    RMSNorm,
    SwiGLU
)


def test_bitnet_config() -> None:
    """Test BitNetConfig initialization and validation."""
    # Test default initialization
    config = BitNetConfig()
    assert config.hidden_size == 768
    assert config.num_attention_heads == 12
    assert config.quantization_bits == 2
    
    # Test custom initialization
    config = BitNetConfig(
        hidden_size=512,
        num_attention_heads=8,
        quantization_bits=3
    )
    assert config.hidden_size == 512
    assert config.num_attention_heads == 8
    assert config.quantization_bits == 3
    
    # Test validation
    with pytest.raises(ValueError):
        BitNetConfig(hidden_size=512, num_attention_heads=7)  # Not divisible
    
    with pytest.raises(ValueError):
        BitNetConfig(quantization_bits=0)  # Invalid bits
    
    with pytest.raises(ValueError):
        BitNetConfig(quantization_method="invalid")  # Invalid method


def test_bitnet_tokenizer() -> None:
    """Test BitNetTokenizer functionality."""
    tokenizer = BitNetTokenizer()
    
    # Test encoding
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 2  # [batch_size, seq_len]
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
    
    # Test special tokens
    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.unk_token_id == 2


def test_bitnet_attention() -> None:
    """Test BitNetAttention functionality."""
    config = BitNetConfig(
        hidden_size=512,
        num_attention_heads=8,
        attention_dropout=0.1
    )
    attention = BitNetAttention(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    encoder_states = torch.randn(batch_size, seq_len * 2, config.hidden_size)
    
    output, attention_weights = attention(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_states
    )
    
    # Check output shape
    assert output.size() == (batch_size, seq_len, config.hidden_size)
    
    # Check attention weights shape
    assert attention_weights.size() == (
        batch_size,
        config.num_attention_heads,
        seq_len,
        seq_len * 2
    )
    
    # Test attention mask
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len * 2)
    attention_mask[:, :, :, seq_len:] = float('-inf')
    
    output, attention_weights = attention(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_states,
        attention_mask=attention_mask
    )
    
    # Check that masked positions have zero attention
    assert torch.all(attention_weights[:, :, :, seq_len:] < 1e-6)


def test_bitnet_transformer() -> None:
    """Test BitNetTransformer functionality."""
    config = BitNetConfig(
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=128
    )
    transformer = BitNetTransformer(config)
    
    # Test encoder forward pass
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = transformer(hidden_states)
    assert output.size() == (batch_size, seq_len, config.hidden_size)
    
    # Test decoder forward pass
    config.is_decoder = True
    decoder = BitNetTransformer(config)
    
    encoder_states = torch.randn(batch_size, seq_len * 2, config.hidden_size)
    output = decoder(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_states
    )
    assert isinstance(output, tuple)
    logits, cache = output
    
    # Check logits shape
    assert logits.size() == (batch_size, seq_len, config.vocab_size)
    
    # Check cache contents
    assert 'past_key_values' in cache
    assert 'attention_weights' in cache
    assert 'cross_attention_weights' in cache


def test_rmsnorm() -> None:
    """Test RMSNorm functionality."""
    hidden_size = 512
    rmsnorm = RMSNorm(hidden_size)
    
    # Test normalization
    x = torch.randn(32, 10, hidden_size)
    output = rmsnorm(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check RMS is approximately 1
    rms = torch.sqrt(torch.mean(output * output, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6)


def test_swiglue() -> None:
    """Test SwiGLU activation."""
    hidden_size = 512
    intermediate_size = 2048
    swiglue = SwiGLU(hidden_size, intermediate_size)
    
    # Test forward pass
    x = torch.randn(32, 10, hidden_size)
    output = swiglue(x)
    
    # Check output shape
    assert output.shape == x.shape


def test_quantization() -> None:
    """Test quantization functionality."""
    config = BitNetConfig(
        hidden_size=512,
        num_attention_heads=8,
        weight_bits=1.58,
        activation_bits=8
    )
    transformer = BitNetTransformer(config)
    
    # Test weight quantization
    weights = torch.randn(100, 100)
    quantized = transformer._absmean_quantization(weights)
    
    # Check values are in {-1, 0, 1}
    unique_values = torch.unique(quantized)
    assert len(unique_values) <= 3
    assert all(v in [-1.0, 0.0, 1.0] for v in unique_values.tolist())
    
    # Test activation quantization
    x = torch.randn(32, 10, 512)
    quantized = transformer._quantize_activations(x)
    
    # Check range
    max_val = 2 ** (config.activation_bits - 1) - 1
    assert torch.all(quantized >= -max_val)
    assert torch.all(quantized <= max_val)


if __name__ == "__main__":
    pytest.main([__file__]) 