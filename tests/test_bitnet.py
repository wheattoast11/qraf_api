"""Tests for BitNet Transformer implementation."""

import pytest
import torch
import numpy as np
from qraf.core.bitnet_transformer import BitNetTransformer, BitNetAttention


def test_bitnet_transformer_initialization():
    """Test BitNet Transformer initialization."""
    config = {
        "hidden_size": 768,
        "num_heads": 12,
        "intermediate_size": 3072,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
    }
    
    transformer = BitNetTransformer(config)
    
    assert transformer is not None
    assert transformer.hidden_size == 768
    assert transformer.num_attention_heads == 12
    assert transformer.intermediate_size == 3072


def test_bitnet_attention_initialization():
    """Test BitNet Attention initialization."""
    attention = BitNetAttention(
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.1,
    )
    
    assert attention is not None
    assert attention.num_attention_heads == 12
    assert attention.attention_head_size == 64  # 768 / 12
    assert attention.all_head_size == 768


def test_bitnet_forward_pass():
    """Test forward pass through BitNet Transformer."""
    config = {
        "hidden_size": 768,
        "num_heads": 12,
    }
    
    transformer = BitNetTransformer(config)
    
    # Create random input tensor
    batch_size = 2
    sequence_length = 10
    input_tensor = torch.randn(batch_size, sequence_length, 768)
    
    # Perform forward pass
    output, attention_probs = transformer(input_tensor)
    
    assert output.shape == input_tensor.shape
    assert attention_probs.shape == (batch_size, 12, sequence_length, sequence_length)


def test_attention_mask():
    """Test attention masking in BitNet Transformer."""
    config = {
        "hidden_size": 768,
        "num_heads": 12,
    }
    
    transformer = BitNetTransformer(config)
    
    # Create input tensor and attention mask
    input_tensor = torch.randn(2, 10, 768)
    attention_mask = torch.zeros(2, 1, 1, 10)
    attention_mask[:, :, :, 5:] = float("-inf")  # Mask out second half
    
    # Forward pass with mask
    output, attention_probs = transformer(input_tensor, attention_mask)
    
    # Check that masked positions have zero attention
    assert torch.all(attention_probs[:, :, :, 5:] < 1e-6)


def test_quantum_gelu():
    """Test quantum-inspired GELU activation."""
    config = {
        "hidden_size": 768,
        "num_heads": 12,
    }
    
    transformer = BitNetTransformer(config)
    
    # Test activation on various inputs
    inputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    outputs = transformer.quantum_gelu(inputs)
    
    # GELU should be monotonically increasing
    assert torch.all(torch.diff(outputs) > 0)
    
    # Check specific values
    assert abs(outputs[2]) < 1e-6  # GELU(0) ≈ 0
    assert outputs[3] > 0.8  # GELU(1) ≈ 0.841
    assert outputs[1] < 0  # GELU(-1) < 0


def test_attention_gradients():
    """Test gradient flow through attention mechanism."""
    attention = BitNetAttention(
        hidden_size=768,
        num_attention_heads=12,
    )
    
    # Create input requiring gradients
    input_tensor = torch.randn(2, 10, 768, requires_grad=True)
    
    # Forward pass
    output, _ = attention(input_tensor)
    
    # Compute gradients
    loss = output.sum()
    loss.backward()
    
    # Check gradient flow
    assert input_tensor.grad is not None
    assert not torch.any(torch.isnan(input_tensor.grad))


def test_transformer_state_dict():
    """Test saving and loading transformer state."""
    config = {
        "hidden_size": 768,
        "num_heads": 12,
    }
    
    # Create and initialize transformer
    transformer1 = BitNetTransformer(config)
    
    # Get state dict
    state_dict = transformer1.state_dict()
    
    # Create new transformer and load state
    transformer2 = BitNetTransformer(config)
    transformer2.load_state_dict(state_dict)
    
    # Compare outputs
    input_tensor = torch.randn(2, 10, 768)
    with torch.no_grad():
        output1, _ = transformer1(input_tensor)
        output2, _ = transformer2(input_tensor)
        
    assert torch.allclose(output1, output2)


def test_attention_weights():
    """Test attention weight analysis."""
    transformer = BitNetTransformer({
        "hidden_size": 768,
        "num_heads": 12,
    })
    
    weights = transformer.get_attention_weights()
    
    assert "query" in weights
    assert "key" in weights
    assert "value" in weights
    
    # Check shapes
    assert weights["query"].shape == (768, 768)
    assert weights["key"].shape == (768, 768)
    assert weights["value"].shape == (768, 768) 