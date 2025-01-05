"""Tests for Claude Integration Interface implementation."""

import pytest
import torch
import numpy as np
import sympy as sp
from qraf.interfaces.claude_augmenter import ClaudeReasoningAugmenter


class MockClaudeInstance:
    """Mock Claude instance for testing."""
    
    def __init__(self):
        self.responses = []
    
    def add_response(self, response):
        self.responses.append(response)


@pytest.fixture
def mock_claude():
    """Create a mock Claude instance."""
    return MockClaudeInstance()


@pytest.fixture
def augmenter(mock_claude):
    """Create a ClaudeReasoningAugmenter instance."""
    config = {
        "transformer_config": {
            "hidden_size": 768,
            "num_heads": 12,
        },
        "sphere_dimensions": 10,
    }
    return ClaudeReasoningAugmenter(mock_claude, config)


def test_augmenter_initialization(augmenter):
    """Test augmenter initialization."""
    assert augmenter.transformer is not None
    assert augmenter.proof_finder is not None
    assert augmenter.knowledge_sphere is not None
    assert len(augmenter.reasoning_memory) == 0
    assert len(augmenter.coherence_history) == 0


def test_query_augmentation(augmenter):
    """Test basic query augmentation."""
    query = "Prove that x^2 + 2x + 1 = (x + 1)^2"
    context = {"domain": "algebra"}
    
    result = augmenter.augment_reasoning(query, context)
    
    assert "original_query" in result
    assert "proof_strategy" in result
    assert "information_density" in result
    assert "quantum_embedding" in result
    assert "reasoning_trace" in result
    
    assert result["original_query"] == query
    assert isinstance(result["information_density"], float)
    assert isinstance(result["quantum_embedding"], np.ndarray)


def test_proof_strategy_generation(augmenter):
    """Test proof strategy generation."""
    query = "x + 0 = x"
    
    result = augmenter.augment_reasoning(query)
    proof_strategy = result["proof_strategy"]
    
    assert "success" in proof_strategy
    assert "proof_path" in proof_strategy
    assert "proof_steps" in proof_strategy


def test_symbolic_conversion(augmenter):
    """Test conversion to symbolic representation."""
    # Test valid mathematical expression
    expr1 = augmenter._convert_to_symbolic("x^2 + 1")
    assert isinstance(expr1, sp.Expr)
    
    # Test invalid expression (fallback case)
    expr2 = augmenter._convert_to_symbolic("not a math expression")
    assert isinstance(expr2, sp.Equality)


def test_query_encoding(augmenter):
    """Test query encoding for transformer."""
    query = "Test query"
    encoded = augmenter._encode_query(query)
    
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 3  # (batch, sequence, features)
    assert encoded.size(0) == 1  # batch size
    assert encoded.size(1) == len(query)  # sequence length
    assert encoded.size(2) == augmenter.transformer.hidden_size


def test_memory_management(augmenter):
    """Test reasoning memory management."""
    # Add multiple queries
    for i in range(5):
        augmenter.augment_reasoning(f"Query {i}")
    
    assert len(augmenter.reasoning_memory) == 5
    assert len(augmenter.coherence_history) == 5
    
    # Test memory limit
    for i in range(100):
        augmenter.augment_reasoning(f"Query {i}")
    
    assert len(augmenter.reasoning_memory) <= 100
    assert len(augmenter.coherence_history) <= 50


def test_reasoning_statistics(augmenter):
    """Test reasoning statistics computation."""
    # Add some test queries
    augmenter.augment_reasoning("x + 0 = x")  # Should succeed
    augmenter.augment_reasoning("x^2 = x")    # Should fail
    
    stats = augmenter.get_reasoning_statistics()
    
    assert "total_queries" in stats
    assert "average_coherence" in stats
    assert "proof_success_rate" in stats
    assert "coherence_trend" in stats
    assert "density_distribution" in stats
    
    assert stats["total_queries"] == 2
    assert 0 <= stats["proof_success_rate"] <= 1
    assert isinstance(stats["coherence_trend"], list)


def test_state_management(augmenter, tmp_path):
    """Test state saving and loading."""
    # Add some test data
    augmenter.augment_reasoning("Test query")
    
    # Save state
    save_path = tmp_path / "augmenter_state.pt"
    augmenter.save_state(str(save_path))
    
    # Create new augmenter and load state
    new_augmenter = ClaudeReasoningAugmenter(MockClaudeInstance())
    new_augmenter.load_state(str(save_path))
    
    # Compare states
    assert len(new_augmenter.reasoning_memory) == len(augmenter.reasoning_memory)
    assert len(new_augmenter.coherence_history) == len(augmenter.coherence_history)


def test_memory_clearing(augmenter):
    """Test memory clearing functionality."""
    # Add some test data
    augmenter.augment_reasoning("Test query")
    assert len(augmenter.reasoning_memory) > 0
    assert len(augmenter.coherence_history) > 0
    
    # Clear memory
    augmenter.clear_memory()
    assert len(augmenter.reasoning_memory) == 0
    assert len(augmenter.coherence_history) == 0


def test_error_handling(augmenter):
    """Test error handling in augmentation."""
    # Test with invalid input
    result = augmenter.augment_reasoning("")
    assert not result["proof_strategy"]["success"]
    
    # Test with malformed query
    result = augmenter.augment_reasoning("@#$%^")
    assert not result["proof_strategy"]["success"]
    assert "error" in result["proof_strategy"]


def test_context_integration(augmenter):
    """Test context integration in reasoning."""
    context = {
        "domain": "algebra",
        "difficulty": "intermediate",
        "previous_steps": ["step1", "step2"],
    }
    
    result = augmenter.augment_reasoning(
        "Prove the Pythagorean theorem",
        context=context,
    )
    
    # Check if context is properly integrated
    assert "context" in result["reasoning_trace"]
    assert result["reasoning_trace"]["context"] == context 