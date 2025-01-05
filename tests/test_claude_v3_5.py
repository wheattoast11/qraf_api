"""Tests for enhanced Claude-3.5 integration."""

import pytest
import torch
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from qraf.interfaces.claude_v3_5 import (
    ClaudeModel,
    ModelCapabilities,
    QuantumEnhancedMessage,
    ClaudeV3_5Augmenter,
)


@pytest.fixture
def mock_anthropic_client():
    """Create mock Anthropic client."""
    client = MagicMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def augmenter(mock_anthropic_client):
    """Create test augmenter instance."""
    with patch("anthropic.Client", return_value=mock_anthropic_client):
        return ClaudeV3_5Augmenter(
            model=ClaudeModel.SONNET,
            api_key="test-key",
        )


def test_model_capabilities():
    """Test model capabilities configuration."""
    sonnet_config = ModelCapabilities(
        context_window=200000,
        max_tokens=4096,
        supports_vision=True,
        supports_parallel=True,
        supports_streaming=True,
        quantum_enhancement_level=1.0,
    )
    
    assert sonnet_config.context_window == 200000
    assert sonnet_config.max_tokens == 4096
    assert sonnet_config.supports_vision
    assert sonnet_config.supports_parallel
    assert sonnet_config.supports_streaming
    assert sonnet_config.quantum_enhancement_level == 1.0


def test_quantum_enhanced_message():
    """Test quantum-enhanced message processing."""
    content = [{"type": "text", "text": "test"}]
    quantum_state = torch.randn(1, 10, 768)
    coherence = 0.8
    
    message = QuantumEnhancedMessage(content, quantum_state, coherence)
    
    assert message.content == content
    assert torch.equal(message.quantum_state, quantum_state)
    assert message.coherence == coherence
    assert len(message.interference_patterns) == 0


def test_quantum_interference():
    """Test quantum interference computation."""
    state1 = torch.randn(1, 10, 768)
    state2 = torch.randn(1, 10, 768)
    
    msg1 = QuantumEnhancedMessage([], state1, 0.8)
    msg2 = QuantumEnhancedMessage([], state2, 0.8)
    
    interference = msg1.apply_quantum_interference(msg2)
    
    assert -1 <= interference <= 1
    assert len(msg1.interference_patterns) == 1


@pytest.mark.asyncio
async def test_generate_response(augmenter, mock_anthropic_client):
    """Test response generation."""
    # Mock Claude response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_anthropic_client.messages.create.return_value = mock_response
    
    # Generate response
    response = await augmenter.generate_response("Test query")
    
    assert "original_response" in response
    assert "quantum_enhanced" in response
    assert "enhanced_content" in response
    
    # Verify quantum enhancement
    quantum_enhanced = response["quantum_enhanced"]
    assert "coherence" in quantum_enhanced
    assert "interference" in quantum_enhanced
    assert "enhancement_factor" in quantum_enhanced


@pytest.mark.asyncio
async def test_streaming_response(augmenter, mock_anthropic_client):
    """Test streaming response generation."""
    # Mock streaming response
    mock_chunks = [
        MagicMock(content=[MagicMock(text="Chunk 1")]),
        MagicMock(content=[MagicMock(text="Chunk 2")]),
    ]
    mock_anthropic_client.messages.create.return_value = mock_chunks
    
    # Generate streaming response
    async for chunk in augmenter.generate_response("Test query", stream=True):
        assert "original_response" in chunk
        assert "quantum_enhanced" in chunk
        assert "enhanced_content" in chunk


def test_context_processing(augmenter):
    """Test context processing."""
    context = {
        "text": "Additional context",
        "images": ["image1.jpg", "image2.jpg"],
    }
    
    blocks = augmenter._process_context(context)
    
    assert len(blocks) == 3  # 1 text + 2 images
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image"
    assert blocks[2]["type"] == "image"


def test_quantum_enhancement(augmenter):
    """Test quantum enhancement process."""
    # Create test response
    response = MagicMock()
    response.content = [MagicMock(text="Test response")]
    
    # Create quantum query
    query_embedding = augmenter._encode_query("Test query")
    quantum_state, coherence = augmenter._compute_quantum_state(query_embedding)
    quantum_query = QuantumEnhancedMessage(
        [{"type": "text", "text": "Test query"}],
        quantum_state,
        coherence,
    )
    
    # Apply enhancement
    enhanced = augmenter._apply_quantum_enhancement(response, quantum_query)
    
    assert isinstance(enhanced["quantum_enhanced"]["coherence"], float)
    assert isinstance(enhanced["quantum_enhanced"]["interference"], float)
    assert enhanced["quantum_enhanced"]["enhancement_factor"] == 1.0


def test_memory_management(augmenter):
    """Test quantum memory management."""
    # Add test responses
    for i in range(105):
        response = {
            "enhanced_content": f"Response {i}",
            "quantum_enhanced": {"coherence": 0.8},
        }
        augmenter._update_quantum_memory(response)
    
    assert len(augmenter.quantum_memory) == 100  # Max size
    assert augmenter.quantum_memory[-1].content[0]["text"] == "Response 104"


def test_proof_integration(augmenter):
    """Test proof integration."""
    proof_result = {
        "success": True,
        "proof_path": ["Step 1", "Step 2", "Step 3"],
    }
    
    content = "Original content"
    enhanced = augmenter._integrate_proof(content, proof_result)
    
    assert "Original content" in enhanced
    assert "Proof steps:" in enhanced
    assert "Step 1" in enhanced
    assert "Step 3" in enhanced


def test_density_integration(augmenter):
    """Test density optimization integration."""
    content = "Test content"
    sphere = {"density": 0.9}
    
    enhanced = augmenter._integrate_density(content, sphere)
    
    assert "[Optimized for clarity]" in enhanced
    assert "Test content" in enhanced


def test_error_handling(augmenter):
    """Test error handling in quantum enhancement."""
    # Test with invalid content
    response = MagicMock()
    response.content = []  # Invalid content
    
    query_embedding = augmenter._encode_query("Test query")
    quantum_state, coherence = augmenter._compute_quantum_state(query_embedding)
    quantum_query = QuantumEnhancedMessage(
        [{"type": "text", "text": "Test query"}],
        quantum_state,
        coherence,
    )
    
    with pytest.raises(Exception):
        augmenter._apply_quantum_enhancement(response, quantum_query) 