"""Tests for Quantum Proof Search implementation."""

import pytest
import sympy as sp
import torch
import numpy as np
from qraf.core.proof_search import QuantumProofPathfinder, QuantumProofState


def test_proof_state_initialization():
    """Test QuantumProofState initialization."""
    x = sp.Symbol("x")
    expr = x**2 + 2*x + 1
    
    state = QuantumProofState(expr)
    
    assert state.expression == expr
    assert state.parent is None
    assert state.rule_applied is None
    assert len(state.children) == 0
    assert state.amplitude == 1.0
    assert state.phase == 0.0


def test_proof_state_equality():
    """Test QuantumProofState equality comparison."""
    x = sp.Symbol("x")
    expr1 = x**2 + 2*x + 1
    expr2 = (x + 1)**2
    expr3 = x**2 + 2*x + 2
    
    state1 = QuantumProofState(expr1)
    state2 = QuantumProofState(expr2)
    state3 = QuantumProofState(expr3)
    
    # These should be equal (symbolically equivalent)
    assert state1 == state2
    
    # These should be different
    assert state1 != state3
    assert state2 != state3


def test_proof_state_path():
    """Test path tracking in proof states."""
    x = sp.Symbol("x")
    expr1 = x**2
    expr2 = x**2 + 2*x
    expr3 = x**2 + 2*x + 1
    
    state1 = QuantumProofState(expr1)
    state2 = state1.add_child(expr2, "addition")
    state3 = state2.add_child(expr3, "addition")
    
    path = state3.get_path_to_root()
    
    assert len(path) == 3
    assert path[0] == state1
    assert path[1] == state2
    assert path[2] == state3
    
    assert path[1].rule_applied == "addition"
    assert path[2].rule_applied == "addition"


def test_proof_pathfinder_initialization():
    """Test QuantumProofPathfinder initialization."""
    pathfinder = QuantumProofPathfinder()
    
    assert pathfinder.transformer is not None
    assert pathfinder.max_depth == 10
    assert pathfinder.num_samples == 1000
    assert len(pathfinder.rules) == 5  # Check number of proof rules


def test_simple_equation_proof():
    """Test proof generation for a simple equation."""
    pathfinder = QuantumProofPathfinder()
    
    # Test x + 0 = x
    x = sp.Symbol("x")
    theorem = sp.Eq(x + 0, x)
    
    proof_report = pathfinder.generate_proof_report(theorem)
    
    assert proof_report["success"] is True
    assert proof_report["proof_found"] is True
    assert len(proof_report["proof_path"]) > 0
    assert len(proof_report["proof_steps"]) > 0


def test_factorization_proof():
    """Test proof generation with factorization."""
    pathfinder = QuantumProofPathfinder()
    
    # Test x^2 + 2x + 1 = (x + 1)^2
    x = sp.Symbol("x")
    theorem = sp.Eq(x**2 + 2*x + 1, (x + 1)**2)
    
    proof_report = pathfinder.generate_proof_report(theorem)
    
    assert proof_report["success"] is True
    assert "factorization" in proof_report["proof_steps"]


def test_expression_encoding():
    """Test symbolic expression encoding."""
    pathfinder = QuantumProofPathfinder()
    
    x = sp.Symbol("x")
    expr = x**2 + 2*x + 1
    
    # Test internal encoding method
    encoded = pathfinder._encode_expression(expr)
    
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 3  # (batch, sequence, features)
    assert encoded.size(0) == 1  # batch size
    assert not torch.any(torch.isnan(encoded))


def test_quantum_interference():
    """Test quantum interference computation."""
    pathfinder = QuantumProofPathfinder()
    
    x = sp.Symbol("x")
    states = [
        QuantumProofState(x**2),
        QuantumProofState(x**2 + 1),
        QuantumProofState(x**2 + 2),
    ]
    
    amplitudes = pathfinder._quantum_interference(states)
    
    assert len(amplitudes) == len(states)
    assert all(0 <= a <= 1 for a in amplitudes)
    assert abs(sum(amplitudes) - 1.0) < 1e-6


def test_proof_rules():
    """Test application of proof rules."""
    pathfinder = QuantumProofPathfinder()
    
    x = sp.Symbol("x")
    state = QuantumProofState(x**2)
    target = x**2 + 1
    visited = {state}
    
    new_states = pathfinder._apply_rules(state, target, visited)
    
    assert len(new_states) > 0
    assert all(isinstance(s, QuantumProofState) for s in new_states)
    assert all(s not in visited for s in new_states)


def test_complex_proof():
    """Test proof generation for a more complex theorem."""
    pathfinder = QuantumProofPathfinder(max_depth=15)
    
    # Test (x + y)^2 = x^2 + 2xy + y^2
    x, y = sp.symbols("x y")
    theorem = sp.Eq((x + y)**2, x**2 + 2*x*y + y**2)
    
    proof_report = pathfinder.generate_proof_report(theorem)
    
    assert proof_report["success"] is True
    assert proof_report["proof_found"] is True
    assert proof_report["depth_reached"] <= 15
    assert len(proof_report["proof_path"]) > 0


def test_invalid_proof():
    """Test proof generation for an invalid theorem."""
    pathfinder = QuantumProofPathfinder()
    
    # Test x^2 = x (invalid for most x)
    x = sp.Symbol("x")
    theorem = sp.Eq(x**2, x)
    
    proof_report = pathfinder.generate_proof_report(theorem)
    
    assert proof_report["success"] is False
    assert proof_report["proof_found"] is False
    assert proof_report["depth_reached"] == pathfinder.max_depth 