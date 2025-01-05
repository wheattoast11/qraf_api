"""Tests for quantization implementation."""

import pytest
import torch
import numpy as np
from qraf.utils.quantization import (
    absmean_quantization,
    compute_information_density,
    QuantizedLinear,
    QuantizedEmbedding,
)


def test_absmean_quantization():
    """Test absmean quantization function."""
    # Create test tensor
    weights = torch.tensor([
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-1.5, -0.5, 0.5, 1.5, 2.5],
    ])
    
    # Test with default bits
    quantized = absmean_quantization(weights)
    assert quantized.shape == weights.shape
    assert torch.all(torch.abs(quantized) <= 2)  # Check range
    
    # Test with different bit settings
    bits_1 = absmean_quantization(weights, bits=1.0)
    bits_2 = absmean_quantization(weights, bits=2.0)
    assert torch.max(torch.abs(bits_1)) < torch.max(torch.abs(bits_2))
    
    # Test different scale methods
    mean_scale = absmean_quantization(weights, scale_method="mean")
    max_scale = absmean_quantization(weights, scale_method="max")
    assert not torch.allclose(mean_scale, max_scale)


def test_information_density():
    """Test information density computation."""
    # Create test tensors
    uniform = torch.ones(100)
    random = torch.randn(100)
    sparse = torch.zeros(100)
    sparse[0] = 1.0
    
    # Compute densities
    uniform_density = compute_information_density(uniform)
    random_density = compute_information_density(random)
    sparse_density = compute_information_density(sparse)
    
    # Check properties
    assert 0 <= uniform_density <= 1
    assert 0 <= random_density <= 1
    assert 0 <= sparse_density <= 1
    assert sparse_density < random_density  # Sparse should have lower density


def test_quantized_linear():
    """Test quantized linear layer."""
    layer = QuantizedLinear(
        in_features=10,
        out_features=5,
        bits=1.58,
        bias=True,
    )
    
    # Test forward pass
    input_tensor = torch.randn(3, 10)  # Batch size 3
    output = layer(input_tensor)
    
    assert output.shape == (3, 5)
    assert not torch.any(torch.isnan(output))
    
    # Test without bias
    layer_no_bias = QuantizedLinear(
        in_features=10,
        out_features=5,
        bias=False,
    )
    output_no_bias = layer_no_bias(input_tensor)
    
    assert output_no_bias.shape == (3, 5)
    assert layer_no_bias.bias is None


def test_quantized_embedding():
    """Test quantized embedding layer."""
    embedding = QuantizedEmbedding(
        num_embeddings=100,
        embedding_dim=10,
        bits=1.58,
        padding_idx=0,
    )
    
    # Test forward pass
    input_indices = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = embedding(input_indices)
    
    assert output.shape == (2, 3, 10)
    assert not torch.any(torch.isnan(output))
    
    # Test padding
    assert torch.all(embedding.weight[0] == 0)  # Padding idx should be zero


def test_gradient_flow():
    """Test gradient flow through quantized layers."""
    layer = QuantizedLinear(5, 3)
    
    # Forward and backward pass
    input_tensor = torch.randn(2, 5, requires_grad=True)
    output = layer(input_tensor)
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert input_tensor.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_quantization_reproducibility():
    """Test reproducibility of quantization."""
    torch.manual_seed(42)
    weights = torch.randn(5, 5)
    
    # Multiple quantizations should be identical
    q1 = absmean_quantization(weights)
    q2 = absmean_quantization(weights)
    
    assert torch.all(q1 == q2)


def test_extreme_values():
    """Test quantization with extreme values."""
    # Test very large values
    large_weights = torch.tensor([1e6, -1e6])
    large_quantized = absmean_quantization(large_weights)
    assert not torch.any(torch.isinf(large_quantized))
    
    # Test very small values
    small_weights = torch.tensor([1e-6, -1e-6])
    small_quantized = absmean_quantization(small_weights)
    assert not torch.any(torch.isnan(small_quantized))


def test_layer_initialization():
    """Test layer initialization."""
    linear = QuantizedLinear(10, 5)
    embedding = QuantizedEmbedding(100, 10)
    
    # Check weight initialization
    assert not torch.any(torch.isnan(linear.weight))
    assert not torch.any(torch.isnan(embedding.weight))
    
    # Check bias initialization
    assert not torch.any(torch.isnan(linear.bias))


def test_string_representation():
    """Test string representation of layers."""
    linear = QuantizedLinear(10, 5, bits=1.58, bias=True)
    embedding = QuantizedEmbedding(100, 10, padding_idx=0)
    
    # Check repr strings
    assert "in_features=10" in str(linear)
    assert "out_features=5" in str(linear)
    assert "num_embeddings=100" in str(embedding)
    assert "embedding_dim=10" in str(embedding)


def test_error_handling():
    """Test error handling in quantization."""
    # Test invalid bits
    with pytest.raises(Exception):
        absmean_quantization(torch.randn(5), bits=0)
    
    # Test invalid scale method
    with pytest.raises(Exception):
        absmean_quantization(torch.randn(5), scale_method="invalid") 