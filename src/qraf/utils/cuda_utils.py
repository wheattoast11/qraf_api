"""CUDA utilities for QRAF."""

from contextlib import contextmanager
from typing import Dict, Optional, Tuple, List, Any, Union
import torch
import torch.nn.functional as F
import cupy as cp
import os
import atexit
import time
from dataclasses import dataclass
import math
import numpy as np
import threading
import statistics

@dataclass
class CUDAConfig:
    """Configuration for CUDA optimizations"""
    max_batch_size: int = 64
    memory_fraction: float = 0.9  # Maximum GPU memory fraction to use
    enable_tf32: bool = True  # Use TensorFloat-32 where available
    enable_cudnn_benchmark: bool = True
    min_chunk_size: int = 1024
    max_workspace_size: int = 1024 * 1024 * 1024  # 1GB
    streams: Dict[str, str] = None  # Stream configuration

    def __post_init__(self):
        if self.streams is None:
            self.streams = {
                "transformer": "high_priority",
                "density": "normal",
                "state": "normal",
                "default": "normal"
            }

class CUDAManager:
    """Manages CUDA resources and optimizations for QRAF components."""
    
    def __init__(self, config: Optional[CUDAConfig] = None):
        """Initialize CUDA manager."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
            
        self.config = config or CUDAConfig()
        self.device = torch.device("cuda")
        
        # Initialize streams with priorities
        self.streams: Dict[str, torch.cuda.Stream] = {}
        for name, priority in self.config.streams.items():
            priority_val = -1 if priority == "high_priority" else 0
            self.streams[name] = torch.cuda.Stream(priority=priority_val)
        
        # Configure CUDA settings
        torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
        torch.backends.cudnn.benchmark = self.config.enable_cudnn_benchmark
        
        # Initialize components
        self.memory_pool = CUDAMemoryPool()
        self.kernels = QRAFCUDAKernels()
        self.profiler = CUDAProfiler()
        
        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.max_error_history = 100
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def optimize_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized attention computation using fused operations."""
        with self.stream_context("transformer"), self.error_context("attention"):
            # Ensure inputs are on GPU and contiguous
            query, key, value = [x.to(self.device).contiguous() for x in [query, key, value]]
            if mask is not None:
                mask = mask.to(self.device).contiguous()
            
            # Fused scaled dot-product attention with automatic mixed precision
            with torch.cuda.amp.autocast():
                scaling = float(query.size(-1)) ** -0.5
                attention_scores = torch.matmul(query * scaling, key.transpose(-2, -1))
                
                if mask is not None:
                    attention_scores = attention_scores + mask
                
                attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
                output = torch.matmul(attention_probs, value)
            
            return output
    
    def optimize_state_compression(
        self,
        state: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        """Memory-efficient state compression."""
        with self.stream_context("state"), self.error_context("compression"):
            state = state.to(self.device).contiguous()
            chunk_size = max(
                min(self.config.max_batch_size, state.size(0)),
                self.config.min_chunk_size
            )
            
            compressed_chunks = []
            for i in range(0, state.size(0), chunk_size):
                chunk = state[i:i + chunk_size]
                U, S, V = torch.svd(chunk)
                compressed = torch.matmul(
                    U[:, :target_size] * S[:target_size],
                    V[:, :target_size].t()
                )
                compressed_chunks.append(compressed)
            
            return torch.cat(compressed_chunks, dim=0)
    
    def optimize_phase_alignment(
        self,
        states: List[torch.Tensor],
        reference_state: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Optimized phase alignment for multiple states."""
        with self.stream_context("state"), self.error_context("phase_alignment"):
            # Move to GPU and optimize memory layout
            states = [s.to(self.device).contiguous() for s in states]
            reference = (reference_state.to(self.device).contiguous() 
                        if reference_state is not None else states[0])
            
            aligned_states = []
            for i in range(0, len(states), self.config.max_batch_size):
                batch = states[i:i + self.config.max_batch_size]
                stacked = torch.stack(batch)
                
                # Compute and apply phase correction
                phase_diff = torch.angle(torch.sum(stacked * reference.conj(), dim=-1))
                correction = torch.exp(1j * phase_diff.unsqueeze(-1))
                aligned = stacked * correction
                
                aligned_states.extend(aligned.unbind(0))
            
            return aligned_states
    
    def optimize_memory_access(
        self,
        states: List[torch.Tensor],
        operation: callable
    ) -> List[torch.Tensor]:
        """Optimize memory access patterns for batch operations."""
        with self.error_context("memory_access"):
            states = [s.to(self.device).contiguous() for s in states]
            total_size = sum(s.numel() * s.element_size() for s in states)
            chunk_size = min(
                self.config.max_batch_size,
                max(1, self.config.max_workspace_size // (total_size // len(states)))
            )
            
            results = []
            for i in range(0, len(states), chunk_size):
                chunk = states[i:i + chunk_size]
                if len(chunk) > 1:
                    result = operation(torch.stack(chunk))
                    results.extend(result.unbind(0))
                else:
                    results.append(operation(chunk[0]))
            
            return results
    
    @contextmanager
    def error_context(self, context: str):
        """Context manager for CUDA error handling."""
        try:
            yield
        except Exception as e:
            error_info = self._log_error(e, context)
            if not self._attempt_recovery(error_info):
                raise RuntimeError(f"CUDA error recovery failed after {self.max_recovery_attempts} attempts") from e
    
    @contextmanager
    def stream_context(self, stream_name: str = "default"):
        """Context manager for CUDA stream."""
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")
        with torch.cuda.stream(self.streams[stream_name]):
            yield
    
    def _log_error(self, error: Exception, context: str, severity: str = "error") -> Dict:
        """Log a CUDA error with context."""
        error_info = {
            "timestamp": time.time(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "severity": severity,
            "memory_stats": self.get_memory_stats(),
            "recovery_attempts": self.recovery_attempts
        }
        
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        return error_info
    
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> bool:
        """Attempt to recover from a CUDA error."""
        if self.recovery_attempts >= self.max_recovery_attempts:
            return False
        
        self.recovery_attempts += 1
        try:
            self.synchronize()
            self.clear_memory()
            self.recovery_attempts = 0
            return True
        except Exception as e:
            self._log_error(e, f"Recovery attempt {self.recovery_attempts} failed", "warning")
            return False
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    
    def synchronize(self, stream_name: Optional[str] = None):
        """Synchronize specified stream or all streams."""
        if stream_name is not None:
            if stream_name not in self.streams:
                raise ValueError(f"Unknown stream: {stream_name}")
            self.streams[stream_name].synchronize()
        else:
            torch.cuda.synchronize()
    
    def clear_memory(self):
        """Clear CUDA memory cache."""
        self.memory_pool.clear()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    def cleanup(self):
        """Release all resources and synchronize streams."""
        print("Cleaning up CUDA resources...")
        self.clear_memory()
        for stream in self.streams.values():
            stream.synchronize()
        print("CUDA resources cleaned up.")

class CUDAMemoryPool:
    """Memory pool for efficient CUDA memory management."""
    
    def __init__(self):
        """Initialize memory pool."""
        # Maintain existing implementation for backward compatibility
        self.reserved_memory: Dict[str, List[torch.Tensor]] = {}
        self.allocation_sizes: Dict[str, int] = {}
        
        # Add new bucketed memory management
        self.memory_buckets: Dict[int, List[torch.Tensor]] = {}
        self.size_multiplier = 2
        self.min_block_size = 256  # 256 bytes minimum
        self.max_block_size = 1024 * 1024 * 1024  # 1GB maximum
        
    def _get_bucket_size(self, size: int) -> int:
        """Get appropriate bucket size for requested memory."""
        bucket_size = self.min_block_size
        while bucket_size < size:
            bucket_size *= self.size_multiplier
            if bucket_size > self.max_block_size:
                return self.max_block_size
        return bucket_size
        
    def allocate(self, size: int, stream_name: str) -> torch.Tensor:
        """Allocate memory from the pool."""
        if stream_name not in self.reserved_memory:
            self.reserved_memory[stream_name] = []
        
        # Try bucketed allocation first
        bucket_size = self._get_bucket_size(size)
        if bucket_size in self.memory_buckets:
            for tensor in self.memory_buckets[bucket_size]:
                if not tensor.is_pinned() and tensor.is_cuda:
                    return tensor.pin_memory()
        
        # Fallback to existing allocation strategy
        for tensor in self.reserved_memory[stream_name]:
            if tensor.numel() * tensor.element_size() >= size and not tensor.is_pinned():
                return tensor.pin_memory()
        
        # Allocate new memory if no suitable tensor is found
        new_tensor = torch.empty(size, dtype=torch.float32, device="cuda")
        self.reserved_memory[stream_name].append(new_tensor)
        
        # Also add to appropriate bucket
        if bucket_size not in self.memory_buckets:
            self.memory_buckets[bucket_size] = []
        self.memory_buckets[bucket_size].append(new_tensor)
        
        return new_tensor.pin_memory()

    def clear_stream_memory(self, stream_name: str):
        """Clear memory associated with a specific stream."""
        if stream_name in self.reserved_memory:
            # Remove tensors from buckets
            for tensor in self.reserved_memory[stream_name]:
                size = tensor.numel() * tensor.element_size()
                bucket_size = self._get_bucket_size(size)
                if bucket_size in self.memory_buckets:
                    self.memory_buckets[bucket_size] = [
                        t for t in self.memory_buckets[bucket_size] if t is not tensor
                    ]
            self.reserved_memory[stream_name].clear()
        
    def clear(self):
        """Clear memory pool."""
        for stream_name in self.reserved_memory:
            self.clear_stream_memory(stream_name)
        self.memory_buckets.clear()
        torch.cuda.empty_cache()

class QRAFCUDAKernels:
    """CUDA kernels for quantum operations"""
    
    def __init__(self):
        self.kernels = {}
        self.stream = cp.cuda.Stream()
        self.compile_kernels()

    def compile_kernels(self):
        """Compile all CUDA kernels"""
        self.kernels["fused_ops"] = self._compile_fused_ops()
        self.kernels["phase_alignment"] = self._compile_phase_alignment()
        self.kernels["density_matrix"] = self._compile_density_matrix()
        self.kernels["quantum_gate"] = self._compile_quantum_gate()
    
    def _compile_fused_ops(self) -> cp.RawKernel:
        """Compile fused operations kernel"""
        kernel_code = r'''
        extern "C" __global__ void fused_ops(
            const float* input,
            float* quantized_output,
            float* density_output,
            int size,
            float scale,
            float coherence_threshold
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                // Shared memory for block-level computations
                __shared__ float block_sum[32];  // For warp-level reduction
                
                // Load input and compute basic values
                float val = input[i];
                float abs_val = abs(val);
                
                // Adaptive gamma calculation based on local statistics
                float local_gamma = scale * (1.0f + abs_val * 0.1f);
                
                // Quantization with error diffusion
                float scaled = abs_val / local_gamma;
                float quantized = round(scaled);
                float error = scaled - quantized;
                
                // Apply error diffusion if not at the edge
                if (i + 1 < size) {
                    atomicAdd(&input[i + 1], error * local_gamma * 0.5f);
                }
                
                if (val < 0) {
                    quantized = -quantized;
                }
                
                // Compute density with coherence preservation
                float density = exp(-abs_val);
                if (density < coherence_threshold) {
                    density *= 0.9f;  // Soft thresholding
                }
                
                // Store results
                quantized_output[i] = quantized;
                density_output[i] = density;
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'fused_ops')
    
    def _compile_phase_alignment(self) -> cp.RawKernel:
        """Compile phase alignment kernel"""
        kernel_code = r'''
        extern "C" __global__ void phase_alignment(
            const float2* states,
            const float2* reference,
            float2* output,
            int size,
            float phase_threshold
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                // Load complex values
                float2 state = states[i];
                float2 ref = reference[i];
                
                // Compute phase difference
                float phase_diff = atan2(
                    state.x * ref.y - state.y * ref.x,
                    state.x * ref.x + state.y * ref.y
                );
                
                // Apply phase correction with threshold
                if (abs(phase_diff) > phase_threshold) {
                    float cos_phase = cos(phase_diff);
                    float sin_phase = sin(phase_diff);
                    
                    output[i].x = state.x * cos_phase - state.y * sin_phase;
                    output[i].y = state.x * sin_phase + state.y * cos_phase;
                } else {
                    output[i] = state;
                }
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'phase_alignment')
    
    def _compile_density_matrix(self) -> cp.RawKernel:
        """Compile density matrix operations kernel"""
        kernel_code = r'''
        extern "C" __global__ void density_matrix(
            const float2* state,
            float2* density,
            int size,
            float trace_threshold
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < size && col < size) {
                // Compute density matrix element
                float2 bra = state[row];
                float2 ket = state[col];
                
                float2 result;
                result.x = bra.x * ket.x + bra.y * ket.y;
                result.y = bra.x * ket.y - bra.y * ket.x;
                
                // Apply trace preservation
                if (row == col) {
                    float trace_element = result.x;
                    if (abs(trace_element - 1.0f) > trace_threshold) {
                        float correction = 1.0f / trace_element;
                        result.x *= correction;
                        result.y *= correction;
                    }
                }
                
                density[row * size + col] = result;
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'density_matrix')
    
    def _compile_quantum_gate(self) -> cp.RawKernel:
        """Compile quantum gate application kernel"""
        kernel_code = r'''
        extern "C" __global__ void quantum_gate(
            const float2* state,
            const float2* gate,
            float2* output,
            int size,
            float fidelity_threshold
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (i < size) {
                float2 result = make_float2(0.0f, 0.0f);
                
                // Apply gate with fidelity checking
                for (int j = 0; j < size; j++) {
                    float2 gate_element = gate[i * size + j];
                    float2 state_element = state[j];
                    
                    // Complex multiplication
                    result.x += gate_element.x * state_element.x 
                              - gate_element.y * state_element.y;
                    result.y += gate_element.x * state_element.y 
                              + gate_element.y * state_element.x;
                }
                
                // Check fidelity
                float fidelity = sqrt(result.x * result.x + result.y * result.y);
                if (fidelity < fidelity_threshold) {
                    // Normalize if fidelity is too low
                    float norm = 1.0f / fidelity;
                    result.x *= norm;
                    result.y *= norm;
                }
                
                output[i] = result;
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'quantum_gate')

    def launch_kernel(
        self,
        kernel_name: str,
        grid_size: Tuple[int, ...],
        block_size: Tuple[int, ...],
        args: Tuple[Any, ...],
        stream: Optional[cp.cuda.Stream] = None
    ):
        """Launch a CUDA kernel with error handling"""
        if kernel_name not in self.kernels:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        
        try:
            with cp.cuda.Stream(stream or self.stream):
                self.kernels[kernel_name](grid_size, block_size, args)
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(f"CUDA kernel '{kernel_name}' failed: {str(e)}")
    
    def _ensure_contiguous(self, array: Union[np.ndarray, torch.Tensor, cp.ndarray]) -> cp.ndarray:
        """Ensure array is a contiguous CuPy array"""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        if isinstance(array, np.ndarray):
            array = cp.asarray(array)
        return cp.ascontiguousarray(array)
    
    def apply_fused_ops(
        self,
        input_tensor: torch.Tensor,
        scale: float = 1.0,
        coherence_threshold: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused quantization and density operations"""
        size = input_tensor.numel()
        threads_per_block = 256
        blocks = (size + threads_per_block - 1) // threads_per_block
        
        # Prepare inputs
        input_cp = self._ensure_contiguous(input_tensor)
        quantized_cp = cp.empty_like(input_cp)
        density_cp = cp.empty_like(input_cp)
        
        # Launch kernel
        self.launch_kernel(
            "fused_ops",
            (blocks,),
            (threads_per_block,),
            (input_cp, quantized_cp, density_cp, size, scale, coherence_threshold)
        )
        
        # Convert back to PyTorch
        return (
            torch.from_numpy(cp.asnumpy(quantized_cp)),
            torch.from_numpy(cp.asnumpy(density_cp))
        )
    
    def apply_phase_alignment(
        self,
        states: torch.Tensor,
        reference: torch.Tensor,
        phase_threshold: float = 0.1
    ) -> torch.Tensor:
        """Apply phase alignment to quantum states"""
        if states.shape[-1] != reference.shape[-1]:
            raise ValueError("States and reference must have same last dimension")
        
        size = states.shape[-1]
        threads_per_block = 256
        blocks = (size + threads_per_block - 1) // threads_per_block
        
        # Prepare inputs
        states_cp = self._ensure_contiguous(states)
        reference_cp = self._ensure_contiguous(reference)
        output_cp = cp.empty_like(states_cp)
        
        # Launch kernel
        self.launch_kernel(
            "phase_alignment",
            (blocks,),
            (threads_per_block,),
            (states_cp, reference_cp, output_cp, size, phase_threshold)
        )
        
        return torch.from_numpy(cp.asnumpy(output_cp))
    
    def apply_density_matrix(
        self,
        state: torch.Tensor,
        trace_threshold: float = 1e-6
    ) -> torch.Tensor:
        """Compute density matrix with trace preservation"""
        size = state.shape[-1]
        threads_per_block = (16, 16)
        blocks = (
            (size + threads_per_block[0] - 1) // threads_per_block[0],
            (size + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        # Prepare inputs
        state_cp = self._ensure_contiguous(state)
        density_cp = cp.empty((size, size), dtype=cp.complex64)
        
        # Launch kernel
        self.launch_kernel(
            "density_matrix",
            blocks,
            threads_per_block,
            (state_cp, density_cp, size, trace_threshold)
        )
        
        return torch.from_numpy(cp.asnumpy(density_cp))
    
    def apply_quantum_gate(
        self,
        state: torch.Tensor,
        gate: torch.Tensor,
        fidelity_threshold: float = 0.99
    ) -> torch.Tensor:
        """Apply quantum gate with fidelity checking"""
        if state.shape[-1] != gate.shape[-1]:
            raise ValueError("State and gate dimensions must match")
        
        size = state.shape[-1]
        threads_per_block = 256
        blocks = (size + threads_per_block - 1) // threads_per_block
        
        # Prepare inputs
        state_cp = self._ensure_contiguous(state)
        gate_cp = self._ensure_contiguous(gate)
        output_cp = cp.empty_like(state_cp)
        
        # Launch kernel
        self.launch_kernel(
            "quantum_gate",
            (blocks,),
            (threads_per_block,),
            (state_cp, gate_cp, output_cp, size, fidelity_threshold)
        )
        
        return torch.from_numpy(cp.asnumpy(output_cp))

class CUDAProfiler:
    """CUDA operation profiler with enhanced memory tracking and NVTX support"""
    
    def __init__(self):
        """Initialize profiler."""
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.timings: Dict[str, float] = {}
        
        # Enhanced tracking
        self.operation_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.memory_snapshots: Dict[str, List[Dict[str, float]]] = {}
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        
        # NVTX color mapping for different operation types
        self.nvtx_colors = {
            "attention": 0xFF0000,  # Red
            "compression": 0x00FF00,  # Green
            "phase": 0x0000FF,      # Blue
            "memory": 0xFFFF00,     # Yellow
            "quantum": 0xFF00FF,    # Purple
            "default": 0xFFFFFF     # White
        }
        
        # Initialize async profiling
        self.async_events: Dict[str, List[torch.cuda.Event]] = {}
        self.async_results: Dict[str, List[float]] = {}
        
    @contextmanager
    def nvtx_range(self, name: str, op_type: str = "default", correlation_id: Optional[int] = None):
        """Context manager for NVTX range with operation type color coding"""
        try:
            torch.cuda.nvtx.range_push(name, color_id=self.nvtx_colors.get(op_type, self.nvtx_colors["default"]))
            if correlation_id is not None:
                torch.cuda.nvtx.mark(f"{name}_{correlation_id}")
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    
    def start_async_profile(self, name: str, stream: Optional[torch.cuda.Stream] = None):
        """Start asynchronous profiling of an operation"""
        if name not in self.async_events:
            self.async_events[name] = []
            
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(stream)
        self.async_events[name].append((start_event, end_event))
    
    def end_async_profile(self, name: str, stream: Optional[torch.cuda.Stream] = None):
        """End asynchronous profiling of an operation"""
        if name not in self.async_events or not self.async_events[name]:
            raise RuntimeError(f"No active async profile for {name}")
            
        start_event, end_event = self.async_events[name][-1]
        end_event.record(stream)
    
    def collect_async_results(self, name: str):
        """Collect results from asynchronous profiling"""
        if name not in self.async_events:
            return []
            
        if name not in self.async_results:
            self.async_results[name] = []
            
        results = []
        for start_event, end_event in self.async_events[name]:
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            results.append(elapsed_time)
            
        self.async_results[name].extend(results)
        self.async_events[name].clear()
        
        return results
    
    @contextmanager
    def profile_operation(
        self,
        name: str,
        op_type: str = "default",
        context: Optional[Dict[str, Any]] = None,
        stream: Optional[torch.cuda.Stream] = None
    ):
        """Profile an operation with NVTX markers and memory tracking"""
        correlation_id = hash(f"{name}_{time.time()}")
        
        try:
            with self.nvtx_range(name, op_type, correlation_id):
                self.start(name, context)
                yield
        finally:
            self.end(name)
            
            # Record additional metrics
            if context is not None:
                context.update({
                    "correlation_id": correlation_id,
                    "op_type": op_type,
                    "stream": str(stream) if stream else "default"
                })
    
    def start(self, name: str, context: Optional[Dict[str, Any]] = None):
        """Start timing an operation with enhanced memory tracking"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        # Enhanced memory snapshot
        memory_snapshot = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
            "fragmentation": self._calculate_fragmentation(),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
        
        if name not in self.memory_snapshots:
            self.memory_snapshots[name] = []
        self.memory_snapshots[name].append(memory_snapshot)
        
        # Store operation info with enhanced context
        self.active_operations[name] = {
            "start_event": start_event,
            "end_event": end_event,
            "start_time": time.time(),
            "context": context or {},
            "start_memory": memory_snapshot,
            "cuda_stream": torch.cuda.current_stream().cuda_stream,
            "thread_id": threading.get_ident()
        }
    
    def end(self, name: str):
        """End timing an operation with enhanced analysis"""
        if name not in self.active_operations:
            return
            
        operation = self.active_operations[name]
        operation["end_event"].record()
        torch.cuda.synchronize()
        
        # Calculate timing
        elapsed_time = operation["start_event"].elapsed_time(operation["end_event"])
        self.timings[name] = elapsed_time
        
        # Enhanced end memory snapshot
        end_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "fragmentation": self._calculate_fragmentation(),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
        
        # Calculate detailed memory changes
        memory_delta = {
            key: end_memory[key] - operation["start_memory"][key]
            for key in end_memory
            if key in operation["start_memory"]
        }
        
        # Enhanced operation record
        operation_record = {
            "name": name,
            "start_time": operation["start_time"],
            "end_time": time.time(),
            "duration_ms": elapsed_time,
            "context": operation["context"],
            "memory": {
                "start": operation["start_memory"],
                "end": end_memory,
                "delta": memory_delta
            },
            "cuda_stream": operation["cuda_stream"],
            "thread_id": operation["thread_id"],
            "performance_metrics": self._calculate_performance_metrics(name)
        }
        
        self.operation_history.append(operation_record)
        if len(self.operation_history) > self.max_history_size:
            self.operation_history.pop(0)
            
        # Cleanup
        del self.active_operations[name]
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio"""
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return 1.0 - (allocated / reserved) if reserved > 0 else 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Estimate cache hit rate based on memory access patterns"""
        # This is a placeholder - actual implementation would depend on
        # hardware-specific metrics
        return 0.0
    
    def _calculate_performance_metrics(self, name: str) -> Dict[str, float]:
        """Calculate detailed performance metrics"""
        if name not in self.timings:
            return {}
            
        # Get memory snapshots for this operation
        snapshots = self.memory_snapshots.get(name, [])
        if not snapshots:
            return {}
            
        # Calculate metrics
        avg_memory = sum(s["allocated"] for s in snapshots) / len(snapshots)
        peak_memory = max(s["allocated"] for s in snapshots)
        memory_volatility = statistics.stdev(s["allocated"] for s in snapshots) if len(snapshots) > 1 else 0
        
        return {
            "avg_memory_gb": avg_memory,
            "peak_memory_gb": peak_memory,
            "memory_volatility": memory_volatility,
            "operations_per_second": 1000 / self.timings[name],  # Based on ms timing
            "memory_efficiency": avg_memory / peak_memory if peak_memory > 0 else 1.0
        }
    
    def get_timings(self) -> Dict[str, float]:
        """Get recorded timings."""
        return self.timings.copy()
    
    def get_memory_profile(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed memory profile for an operation or all operations."""
        if name is not None and name in self.memory_snapshots:
            snapshots = self.memory_snapshots[name]
            return {
                "snapshots": snapshots,
                "peak_usage": max(s["allocated"] for s in snapshots),
                "average_usage": sum(s["allocated"] for s in snapshots) / len(snapshots),
                "fragmentation_trend": [s["fragmentation"] for s in snapshots],
                "cache_hit_rates": [s["cache_hit_rate"] for s in snapshots]
            }
        
        return {
            name: self.get_memory_profile(name)
            for name in self.memory_snapshots
        }
    
    def get_operation_history(
        self,
        name: Optional[str] = None,
        op_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered operation history."""
        history = self.operation_history
        
        if name is not None:
            history = [op for op in history if op["name"] == name]
        if op_type is not None:
            history = [op for op in history if op["context"].get("op_type") == op_type]
        if start_time is not None:
            history = [op for op in history if op["start_time"] >= start_time]
        if end_time is not None:
            history = [op for op in history if op["end_time"] <= end_time]
        if limit is not None:
            history = history[-limit:]
            
        return history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "overall_metrics": {
                "total_operations": len(self.operation_history),
                "total_duration_ms": sum(op["duration_ms"] for op in self.operation_history),
                "peak_memory_gb": max(op["memory"]["end"]["allocated"] for op in self.operation_history),
                "average_operation_time_ms": statistics.mean(op["duration_ms"] for op in self.operation_history)
            },
            "operation_types": {
                op_type: {
                    "count": len([op for op in self.operation_history if op["context"].get("op_type") == op_type]),
                    "avg_duration_ms": statistics.mean(op["duration_ms"] for op in self.operation_history if op["context"].get("op_type") == op_type),
                    "peak_memory_gb": max(op["memory"]["end"]["allocated"] for op in self.operation_history if op["context"].get("op_type") == op_type)
                }
                for op_type in self.nvtx_colors.keys()
            },
            "memory_trends": {
                "fragmentation": [op["memory"]["end"]["fragmentation"] for op in self.operation_history],
                "cache_hit_rates": [op["memory"]["end"]["cache_hit_rate"] for op in self.operation_history]
            }
        }
    
    def clear(self):
        """Clear all profiling data."""
        self.timings.clear()
        self.operation_history.clear()
        self.memory_snapshots.clear()
        self.active_operations.clear()
        self.async_events.clear()
        self.async_results.clear()
        torch.cuda.reset_peak_memory_stats() 