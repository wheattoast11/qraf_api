"""Tests for density optimization implementation."""

import pytest
import numpy as np
import torch
import networkx as nx
from qraf.utils.density_optimization import DensityNormalizedSphere


def test_sphere_initialization():
    """Test DensityNormalizedSphere initialization."""
    sphere = DensityNormalizedSphere(
        dimensions=10,
        density_threshold=0.5,
        learning_rate=0.01,
    )
    
    assert sphere.dimensions == 10
    assert sphere.density_threshold == 0.5
    assert sphere.learning_rate == 0.01
    assert isinstance(sphere.density_graph, nx.Graph)
    assert len(sphere.density_history) == 0


def test_numeric_conversion():
    """Test conversion of different data types to numeric representation."""
    sphere = DensityNormalizedSphere(dimensions=5)
    
    # Test numpy array
    np_data = np.array([1, 2, 3, 4, 5])
    np_result = sphere._convert_to_numeric(np_data)
    assert isinstance(np_result, np.ndarray)
    assert np.array_equal(np_result, np_data)
    
    # Test torch tensor
    torch_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    torch_result = sphere._convert_to_numeric(torch_data)
    assert isinstance(torch_result, np.ndarray)
    assert np.array_equal(torch_result, torch_data.numpy())
    
    # Test list
    list_data = [1, 2, 3, 4, 5]
    list_result = sphere._convert_to_numeric(list_data)
    assert isinstance(list_result, np.ndarray)
    assert np.array_equal(list_result, np.array(list_data))
    
    # Test string
    str_data = "hello"
    str_result = sphere._convert_to_numeric(str_data)
    assert isinstance(str_result, np.ndarray)
    assert len(str_result) == 5
    assert np.array_equal(str_result, np.array([ord(c) for c in str_data]))
    
    # Test scalar
    scalar_data = 42
    scalar_result = sphere._convert_to_numeric(scalar_data)
    assert isinstance(scalar_result, np.ndarray)
    assert len(scalar_result) == 5
    assert np.all(scalar_result == 42)


def test_density_computation():
    """Test information density computation."""
    sphere = DensityNormalizedSphere()
    
    # Test uniform distribution (maximum entropy)
    uniform_data = np.ones(100)
    uniform_density = sphere._compute_density(uniform_data)
    assert uniform_density > 0.99  # Should be close to 1
    
    # Test single value (minimum entropy)
    single_data = np.zeros(100)
    single_density = sphere._compute_density(single_data)
    assert single_density < 0.01  # Should be close to 0
    
    # Test random distribution
    random_data = np.random.randn(100)
    random_density = sphere._compute_density(random_data)
    assert 0 <= random_density <= 1


def test_sphere_creation():
    """Test creation of information sphere."""
    sphere = DensityNormalizedSphere()
    
    data = np.random.randn(10)
    metadata = {"type": "test", "source": "unit_test"}
    
    result = sphere.create(data, metadata)
    
    assert "center" in result
    assert "radius" in result
    assert "density" in result
    assert "embedding" in result
    assert "metadata" in result
    assert result["metadata"] == metadata


def test_density_optimization():
    """Test density optimization process."""
    sphere = DensityNormalizedSphere()
    
    # Create initial sphere with low density
    initial_data = np.zeros(10)  # Low entropy data
    initial_sphere = sphere.create(initial_data)
    
    # Optimize density
    target_density = 0.8
    optimized_sphere = sphere.optimize_density(
        initial_sphere,
        target_density=target_density,
    )
    
    assert optimized_sphere["density"] > initial_sphere["density"]
    assert abs(optimized_sphere["density"] - target_density) < 0.2


def test_sphere_merging():
    """Test merging of information spheres."""
    sphere = DensityNormalizedSphere()
    
    # Create two spheres
    data1 = np.random.randn(10)
    data2 = np.random.randn(10)
    
    sphere1 = sphere.create(data1, {"type": "A"})
    sphere2 = sphere.create(data2, {"type": "B"})
    
    # Merge spheres
    merged = sphere.merge_spheres(sphere1, sphere2)
    
    assert len(merged["embedding"]) == len(data1) + len(data2)
    assert merged["metadata"]["type"] == "B"  # Last one overwrites
    assert isinstance(merged["density"], float)


def test_similarity_computation():
    """Test similarity computation between embeddings."""
    sphere = DensityNormalizedSphere()
    
    # Identical embeddings
    embedding1 = np.array([1, 2, 3])
    similarity = sphere._compute_similarity(embedding1, embedding1)
    assert np.isclose(similarity, 1.0)
    
    # Orthogonal embeddings
    embedding2 = np.array([0, 0, 1])
    embedding3 = np.array([1, 0, 0])
    similarity = sphere._compute_similarity(embedding2, embedding3)
    assert np.isclose(similarity, 0.0)
    
    # Similar embeddings
    embedding4 = np.array([1, 1, 1])
    embedding5 = np.array([1, 1, 0.9])
    similarity = sphere._compute_similarity(embedding4, embedding5)
    assert 0.9 < similarity < 1.0


def test_graph_statistics():
    """Test knowledge graph statistics computation."""
    sphere = DensityNormalizedSphere()
    
    # Add some test data
    for _ in range(5):
        sphere.create(np.random.randn(10))
    
    stats = sphere.get_graph_statistics()
    
    assert stats["num_nodes"] == 5
    assert "num_edges" in stats
    assert "average_density" in stats
    assert "density_distribution" in stats


def test_density_history():
    """Test density history tracking."""
    sphere = DensityNormalizedSphere()
    
    # Create multiple spheres
    for _ in range(3):
        sphere.create(np.random.randn(10))
    
    history = sphere.get_density_history()
    
    assert isinstance(history, list)
    assert len(history) <= 3  # Should match number of creations


def test_error_handling():
    """Test error handling in density optimization."""
    sphere = DensityNormalizedSphere()
    
    # Test invalid data type
    with pytest.raises(ValueError):
        sphere._convert_to_numeric({"invalid": "data"})
    
    # Test empty data
    empty_data = np.array([])
    empty_sphere = sphere.create(empty_data)
    assert empty_sphere["density"] == 0 