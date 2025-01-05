"""Density optimization utilities for quantum reasoning."""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from torch import Tensor
from ..utils.cuda_utils import CUDAManager, CUDAProfiler, QRAFCUDAKernels


class DensityOptimizer:
    """Optimizes density distributions with CUDA acceleration."""
    
    def __init__(
        self,
        density_threshold: float = 0.7,
        learning_rate: float = 0.01,
        cuda_manager: Optional['CUDAManager'] = None
    ):
        """Initialize density optimizer.
        
        Args:
            density_threshold: Target density threshold
            learning_rate: Learning rate for optimization
            cuda_manager: Optional CUDA manager for GPU acceleration
        """
        self.density_threshold = density_threshold
        self.learning_rate = learning_rate
        
        # CUDA support
        self.cuda_manager = cuda_manager
        self.device = cuda_manager.device if cuda_manager else torch.device('cpu')
        if cuda_manager and cuda_manager.is_available():
            self.kernels = QRAFCUDAKernels()
            self.profiler = CUDAProfiler()
            self.use_fused_ops = True
        else:
            self.use_fused_ops = False
            
    def _compute_density_tensor(
        self,
        embedding: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Compute density with CUDA optimization."""
        if self.use_fused_ops:
            with self.profiler.profile_operation("compute_density", "quantum"):
                # Compute pairwise distances efficiently
                dists = torch.cdist(embedding, embedding)
                
                # Apply Gaussian kernel with fused operation
                sigma = torch.std(dists)
                density = torch.exp(-dists / (2 * sigma * sigma))
                
                # Normalize with stability
                density = density / (torch.sum(density, dim=-1, keepdim=True) + epsilon)
                
                return density.mean()
        else:
            # Fallback to standard computation
            dists = torch.cdist(embedding, embedding)
            sigma = torch.std(dists)
            density = torch.exp(-dists / (2 * sigma * sigma))
            density = density / (torch.sum(density, dim=-1, keepdim=True) + epsilon)
            return density.mean()
            
    def _compute_density(
        self,
        embedding: torch.Tensor,
        batch_size: int = 1024
    ) -> float:
        """Compute density with batch processing."""
        if not embedding.is_cuda:
            embedding = embedding.to(self.device)
            
        total_density = 0.0
        num_batches = 0
        
        for i in range(0, embedding.size(0), batch_size):
            batch = embedding[i:i + batch_size]
            density = self._compute_density_tensor(batch)
            total_density += density.item()
            num_batches += 1
            
        return total_density / num_batches
            
    def _optimize_density_impl(
        self,
        sphere: Dict[str, Any],
        target_density: Optional[float] = None,
        batch_size: int = 1024
    ) -> Dict[str, Any]:
        """Implementation of density optimization with CUDA support."""
        if self.cuda_manager:
            with self.cuda_manager.error_context("DensityOptimizer._optimize_density_impl"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._optimize_density_cuda(sphere, target_density, batch_size)
        else:
            return self._optimize_density_cpu(sphere, target_density, batch_size)
            
    def _optimize_density_cuda(
        self,
        sphere: Dict[str, Any],
        target_density: Optional[float] = None,
        batch_size: int = 1024
    ) -> Dict[str, Any]:
        """CUDA-optimized density optimization."""
        target = target_density or self.density_threshold
        
        # Move embedding to GPU and prepare for optimization
        embedding = torch.tensor(
            sphere['embedding'],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        
        # Process in batches
        for start_idx in range(0, embedding.size(0), batch_size):
            end_idx = min(start_idx + batch_size, embedding.size(0))
            batch = embedding[start_idx:end_idx]
            
            # Compute density with CUDA optimization
            density = self._compute_density_tensor(batch)
            loss = torch.abs(density - target)
            
            if loss.item() < 1e-4:
                break
                
            # Optimize with CUDA acceleration
            loss.backward()
            with torch.no_grad():
                batch -= self.learning_rate * batch.grad
                batch.grad.zero_()
                
            embedding[start_idx:end_idx] = batch
            
        # Update sphere with optimized values
        optimized_sphere = sphere.copy()
        optimized_sphere.update({
            'embedding': embedding.detach().cpu().numpy(),
            'density': self._compute_density(embedding),
            'center': torch.mean(embedding, dim=0).detach().cpu().numpy(),
            'radius': torch.std(embedding).item(),
        })
        
        return optimized_sphere
            
    def _optimize_density_cpu(
        self,
        sphere: Dict[str, Any],
        target_density: Optional[float] = None,
        batch_size: int = 1024
    ) -> Dict[str, Any]:
        """CPU fallback for density optimization."""
        target = target_density or self.density_threshold
        
        # Convert to tensor
        embedding = torch.tensor(
            sphere['embedding'],
            dtype=torch.float32,
            requires_grad=True
        )
        
        # Simple gradient descent
        for _ in range(100):
            density = self._compute_density_tensor(embedding)
            loss = torch.abs(density - target)
            
            if loss.item() < 1e-4:
                break
                
            loss.backward()
            with torch.no_grad():
                embedding -= self.learning_rate * embedding.grad
                embedding.grad.zero_()
        
        # Update sphere
        optimized_sphere = sphere.copy()
        optimized_sphere.update({
            'embedding': embedding.detach().numpy(),
            'density': self._compute_density(embedding),
            'center': torch.mean(embedding, dim=0).detach().numpy(),
            'radius': torch.std(embedding).item(),
        })
        
        return optimized_sphere
    
    def create(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a density-normalized representation of input data.
        
        Args:
            data: Input data to process
            metadata: Optional metadata for the sphere
            
        Returns:
            Density-normalized information sphere
        """
        if self.cuda_manager:
            with self.cuda_manager.error_context("DensityOptimizer.create"):
                with self.cuda_manager.stream_context("density"):
                    return self._create_impl(data, metadata)
        else:
            return self._create_impl(data, metadata)
            
    def _create_impl(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Implementation of create method."""
        # Convert input to numeric representation
        numeric_rep = self._convert_to_numeric(data)
        
        # Move to appropriate device
        if isinstance(numeric_rep, np.ndarray):
            numeric_rep = torch.from_numpy(numeric_rep).to(self.device)
        elif isinstance(numeric_rep, torch.Tensor):
            numeric_rep = numeric_rep.to(self.device)
        
        # Compute information density
        density = self._compute_density(numeric_rep)
        
        # Create spherical embedding
        sphere = {
            'center': torch.mean(numeric_rep, dim=0).cpu().numpy(),
            'radius': torch.std(numeric_rep).item(),
            'density': density,
            'embedding': numeric_rep.cpu().numpy(),
            'metadata': metadata or {},
        }
        
        # Add to knowledge graph
        self._update_knowledge_graph(sphere)
        
        return sphere
    
    def optimize_density(
        self,
        sphere: Dict[str, Any],
        target_density: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Optimize the density of an information sphere.
        
        Args:
            sphere: Information sphere to optimize
            target_density: Optional target density value
            
        Returns:
            Optimized information sphere
        """
        if self.cuda_manager:
            with self.cuda_manager.error_context("DensityOptimizer.optimize_density"):
                with self.cuda_manager.stream_context("density"):
                    return self._optimize_density_impl(sphere, target_density)
        else:
            return self._optimize_density_impl(sphere, target_density)
    
    def merge_spheres(
        self,
        sphere1: Dict[str, Any],
        sphere2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge two information spheres.
        
        Args:
            sphere1: First information sphere
            sphere2: Second information sphere
            
        Returns:
            Merged information sphere
        """
        # Combine embeddings
        combined_embedding = np.concatenate([
            sphere1['embedding'],
            sphere2['embedding'],
        ])
        
        # Create merged sphere
        merged = self.create(combined_embedding)
        
        # Combine metadata
        merged['metadata'] = {
            **sphere1.get('metadata', {}),
            **sphere2.get('metadata', {}),
        }
        
        return merged
    
    def _convert_to_numeric(
        self,
        data: Any,
    ) -> np.ndarray:
        """
        Convert input to numeric representation.
        
        Args:
            data: Input data
            
        Returns:
            Numeric representation
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().numpy()
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        elif isinstance(data, str):
            # Convert string to character codes
            return np.array([ord(c) for c in data[:self.dimensions]])
        elif isinstance(data, (int, float)):
            return np.array([data] * self.dimensions)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _update_knowledge_graph(
        self,
        sphere: Dict[str, Any],
    ) -> None:
        """
        Update knowledge graph with new information sphere.
        
        Args:
            sphere: Information sphere to add
        """
        # Create unique identifier for the sphere
        sphere_id = hash(tuple(sphere['center']))
        
        # Add node to graph
        self.density_graph.add_node(
            sphere_id,
            **sphere,
        )
        
        # Connect to similar spheres
        for node in self.density_graph.nodes:
            if node == sphere_id:
                continue
                
            other_sphere = self.density_graph.nodes[node]
            similarity = self._compute_similarity(
                sphere['embedding'],
                other_sphere['embedding'],
            )
            
            if similarity > 0.5:  # Threshold for connection
                self.density_graph.add_edge(
                    sphere_id,
                    node,
                    weight=similarity,
                )
    
    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        # Ensure same shape
        min_dim = min(len(embedding1), len(embedding2))
        e1 = embedding1[:min_dim]
        e2 = embedding2[:min_dim]
        
        # Compute cosine similarity
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def get_density_history(self) -> List[float]:
        """Get history of density values."""
        return self.density_history
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            'num_nodes': self.density_graph.number_of_nodes(),
            'num_edges': self.density_graph.number_of_edges(),
            'average_density': np.mean([
                data['density']
                for _, data in self.density_graph.nodes(data=True)
            ]),
            'density_distribution': np.histogram(
                [data['density'] for _, data in self.density_graph.nodes(data=True)],
                bins='auto',
            )[0].tolist(),
        } 