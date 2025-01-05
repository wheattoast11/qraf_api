"""Quantization utilities for efficient tensor operations."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.cuda_utils import CUDAManager, CUDAProfiler, QRAFCUDAKernels


def absmean_quantization(
    weights: torch.Tensor,
    bits: float = 1.58,
    scale_method: str = "mean",
    cuda_manager: Optional['CUDAManager'] = None,
    batch_size: int = 1024
) -> torch.Tensor:
    """
    Implement flexible quantization with CUDA optimization.
    
    Args:
        weights: Input weight tensor
        bits: Quantization bit representation (default: 1.58)
        scale_method: Method to compute scaling factor ("mean" or "max")
        cuda_manager: Optional CUDA manager for GPU acceleration
        batch_size: Batch size for processing
        
    Returns:
        Quantized weights
    """
    device = cuda_manager.device if cuda_manager else torch.device('cpu')
    use_cuda = cuda_manager and cuda_manager.is_available()
    
    # Move weights to appropriate device
    weights = weights.to(device)
    
    if use_cuda:
        with cuda_manager.error_context("absmean_quantization"):
            with cuda_manager.stream_context("quantization"):
                return _absmean_quantization_cuda(
                    weights, bits, scale_method, cuda_manager, batch_size
                )
    else:
        return _absmean_quantization_cpu(weights, bits, scale_method)


def _absmean_quantization_cuda(
    weights: torch.Tensor,
    bits: float,
    scale_method: str,
    cuda_manager: 'CUDAManager',
    batch_size: int
) -> torch.Tensor:
    """CUDA-optimized implementation of absmean quantization."""
    kernels = QRAFCUDAKernels()
    profiler = CUDAProfiler()
    
    with profiler.profile_operation("quantization", "cuda"):
        # Process in batches
        quantized_chunks = []
        for i in range(0, weights.size(0), batch_size):
            chunk = weights[i:i + batch_size]
            
            # Compute scaling factor
            if scale_method == "mean":
                gamma = torch.mean(torch.abs(chunk))
            else:  # "max"
                gamma = torch.max(torch.abs(chunk))
            
            # Compute number of quantization levels
            levels = 2 ** bits - 1
            
            # Quantize with CUDA optimization
            quantized = torch.clamp(
                torch.round(chunk / (gamma + 1e-8) * (levels / 2)),
                min=-levels/2,
                max=levels/2,
            )
            
            quantized_chunks.append(quantized)
        
        return torch.cat(quantized_chunks, dim=0)


def _absmean_quantization_cpu(
    weights: torch.Tensor,
    bits: float,
    scale_method: str
) -> torch.Tensor:
    """CPU implementation of absmean quantization."""
    # Compute scaling factor
    if scale_method == "mean":
        gamma = torch.mean(torch.abs(weights))
    else:  # "max"
        gamma = torch.max(torch.abs(weights))
    
    # Compute number of quantization levels
    levels = 2 ** bits - 1
    
    # Clip and quantize
    quantized = torch.clamp(
        torch.round(weights / (gamma + 1e-8) * (levels / 2)),
        min=-levels/2,
        max=levels/2,
    )
    
    return quantized


def compute_information_density(
    tensor: torch.Tensor,
    epsilon: float = 1e-8,
    cuda_manager: Optional['CUDAManager'] = None,
    batch_size: int = 1024
) -> float:
    """
    Calculate information density with CUDA optimization.
    
    Args:
        tensor: Input tensor
        epsilon: Small constant for numerical stability
        cuda_manager: Optional CUDA manager for GPU acceleration
        batch_size: Batch size for processing
        
    Returns:
        Information density metric
    """
    device = cuda_manager.device if cuda_manager else torch.device('cpu')
    use_cuda = cuda_manager and cuda_manager.is_available()
    
    # Move tensor to appropriate device
    tensor = tensor.to(device)
    
    if use_cuda:
        with cuda_manager.error_context("compute_information_density"):
            with cuda_manager.stream_context("quantization"):
                return _compute_information_density_cuda(
                    tensor, epsilon, cuda_manager, batch_size
                )
    else:
        return _compute_information_density_cpu(tensor, epsilon)


def _compute_information_density_cuda(
    tensor: torch.Tensor,
    epsilon: float,
    cuda_manager: 'CUDAManager',
    batch_size: int
) -> float:
    """CUDA-optimized implementation of information density computation."""
    kernels = QRAFCUDAKernels()
    profiler = CUDAProfiler()
    
    with profiler.profile_operation("density_computation", "cuda"):
        # Process in batches
        total_density = 0.0
        num_batches = 0
        
        for i in range(0, tensor.size(0), batch_size):
            chunk = tensor[i:i + batch_size]
            
            # Flatten chunk
            flat_chunk = chunk.reshape(-1)
            
            # Compute probabilities with CUDA optimization
            probs = F.softmax(flat_chunk, dim=0)
            
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log2(probs + epsilon))
            
            total_density += entropy.item()
            num_batches += 1
        
        return total_density / num_batches


def _compute_information_density_cpu(
    tensor: torch.Tensor,
    epsilon: float
) -> float:
    """CPU implementation of information density computation."""
    # Flatten tensor
    flat_tensor = tensor.flatten()
    
    # Compute probability distribution
    probabilities = F.softmax(flat_tensor, dim=0)
    
    # Calculate entropy
    entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon))
    
    return entropy.item()


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights and CUDA optimization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: float = 1.58,
        bias: bool = True,
        scale_method: str = "mean",
        cuda_manager: Optional['CUDAManager'] = None
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bits: Number of bits for quantization
            bias: Whether to include bias
            scale_method: Method to compute scaling factor
            cuda_manager: Optional CUDA manager for GPU acceleration
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.scale_method = scale_method
        self.cuda_manager = cuda_manager
        self.device = cuda_manager.device if cuda_manager else torch.device('cpu')
        
        # Initialize weights and optional bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Move tensors to appropriate device
        x = x.to(self.device)
        weight = self.weight.to(self.device)
        
        # Quantize weights
        quantized_weight = absmean_quantization(
            weight,
            bits=self.bits,
            scale_method=self.scale_method,
            cuda_manager=self.cuda_manager
        )
        
        # Compute output
        output = F.linear(x, quantized_weight)
        if self.bias is not None:
            output = output + self.bias
            
        return output


class QuantizedEmbedding(nn.Module):
    """Embedding layer with quantized weights and CUDA optimization."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: float = 1.58,
        padding_idx: Optional[int] = None,
        scale_method: str = "mean",
        cuda_manager: Optional['CUDAManager'] = None
    ):
        """
        Initialize quantized embedding layer.
        
        Args:
            num_embeddings: Size of the dictionary
            embedding_dim: Size of each embedding vector
            bits: Number of bits for quantization
            padding_idx: Index for padding token
            scale_method: Method to compute scaling factor
            cuda_manager: Optional CUDA manager for GPU acceleration
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        self.padding_idx = padding_idx
        self.scale_method = scale_method
        self.cuda_manager = cuda_manager
        self.device = cuda_manager.device if cuda_manager else torch.device('cpu')
        
        # Initialize embedding weights
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Move tensors to appropriate device
        x = x.to(self.device)
        weight = self.weight.to(self.device)
        
        # Quantize weights
        quantized_weight = absmean_quantization(
            weight,
            bits=self.bits,
            scale_method=self.scale_method,
            cuda_manager=self.cuda_manager
        )
        
        # Compute embeddings
        return F.embedding(
            x,
            quantized_weight,
            self.padding_idx,
            None,
            2,
            False,
            False
        )
    
    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"bits={self.bits}, "
            f"padding_idx={self.padding_idx}, "
            f"scale_method={self.scale_method}"
        ) 