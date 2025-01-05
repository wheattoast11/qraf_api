"""Network of specialized Claude instances with BitTensor state management."""

from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
import math

from .state_network import StateNode, QuantumStateNetwork
from .bitnet_transformer import BitNetTransformer
from ..interfaces.claude_v3_5 import ClaudeV3_5Augmenter
from ..utils.cuda_utils import CUDAManager, CUDAProfiler, QRAFCUDAKernels


@dataclass
class SpecializedClaudeInstance:
    """Represents a specialized Claude instance with its quantum and BitTensor states."""
    
    instance_id: int
    specialization: str  # Type of specialization (e.g., "Abstract_Reasoning")
    claude: ClaudeV3_5Augmenter
    state_node: StateNode
    subspace_range: Tuple[int, int]  # Range in the 768-dim space
    coherence_threshold: float = 0.7  # Coherence threshold for this instance
    phase_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    cuda_manager: Optional['CUDAManager'] = None
    
    def __post_init__(self):
        """Initialize CUDA support after instance creation."""
        self.device = self.cuda_manager.device if self.cuda_manager else torch.device('cpu')
        if self.cuda_manager and self.cuda_manager.is_available():
            self.kernels = QRAFCUDAKernels()
            self.profiler = CUDAProfiler()
            self.use_cuda = True
            
            # Move specialization mask to GPU
            self.specialization_mask = self._create_specialization_mask().to(self.device)
        else:
            self.use_cuda = False
            self.specialization_mask = self._create_specialization_mask()
    
    def update_metrics(self, phase: float, coherence: float):
        """Update phase and coherence history with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("update_metrics"):
                with self.cuda_manager.stream_context("quantum"):
                    self._update_metrics_cuda(phase, coherence)
        else:
            self._update_metrics_cpu(phase, coherence)
    
    def _update_metrics_cuda(self, phase: float, coherence: float):
        """CUDA-optimized metrics update."""
        with self.profiler.profile_operation("metrics_update", "cuda"):
            # Update histories with CUDA acceleration
            if len(self.phase_history) > 1000:  # Maintain history size
                self.phase_history = self.phase_history[-1000:]
                self.coherence_history = self.coherence_history[-1000:]
            
            # Add new metrics with CUDA optimization
            phase_tensor = torch.tensor([phase], device=self.device)
            coherence_tensor = torch.tensor([coherence], device=self.device)
            
            # Apply quantum-aware smoothing
            if self.phase_history:
                last_phase = torch.tensor([self.phase_history[-1]], device=self.device)
                phase_diff = torch.abs(phase_tensor - last_phase)
                if phase_diff > math.pi:
                    phase_tensor = last_phase + torch.sign(phase_tensor - last_phase) * phase_diff
            
            self.phase_history.append(phase_tensor.item())
            self.coherence_history.append(coherence_tensor.item())
    
    def _update_metrics_cpu(self, phase: float, coherence: float):
        """CPU implementation of metrics update."""
        if len(self.phase_history) > 1000:  # Maintain history size
            self.phase_history = self.phase_history[-1000:]
            self.coherence_history = self.coherence_history[-1000:]
            
        # Add new metrics
        self.phase_history.append(phase)
        self.coherence_history.append(coherence)
    
    def get_average_coherence(self) -> float:
        """Get average coherence over history with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("average_coherence"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._get_average_coherence_cuda()
        else:
            return self._get_average_coherence_cpu()
    
    def _get_average_coherence_cuda(self) -> float:
        """CUDA-optimized coherence averaging."""
        with self.profiler.profile_operation("coherence_averaging", "cuda"):
            if not self.coherence_history:
                return 0.0
            
            # Convert history to tensor and move to GPU
            coherence_tensor = torch.tensor(
                self.coherence_history,
                device=self.device
            )
            
            # Apply exponential weighting to recent values
            weights = torch.exp(torch.linspace(
                0, 1, len(self.coherence_history),
                device=self.device
            ))
            weights = weights / weights.sum()
            
            # Compute weighted average
            return (coherence_tensor * weights).sum().item()
    
    def _get_average_coherence_cpu(self) -> float:
        """CPU implementation of coherence averaging."""
        if not self.coherence_history:
            return 0.0
        return sum(self.coherence_history) / len(self.coherence_history)
    
    def project_to_subspace(self, state: torch.Tensor) -> torch.Tensor:
        """Project state to instance's specialized subspace with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("subspace_projection"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._project_to_subspace_cuda(state)
        else:
            return self._project_to_subspace_cpu(state)
    
    def _project_to_subspace_cuda(self, state: torch.Tensor) -> torch.Tensor:
        """CUDA-optimized subspace projection."""
        with self.profiler.profile_operation("subspace_projection", "cuda"):
            # Move state to GPU if needed
            if not state.is_cuda:
                state = state.to(self.device)
            
            # Apply specialization mask with CUDA optimization
            projected_state = self.kernels.apply_specialization_mask(
                state,
                self.specialization_mask
            )
            
            # Apply phase-preserving projection
            if self.phase_history:
                recent_phase = torch.tensor(
                    self.phase_history[-1],
                    device=self.device
                )
                projected_state = self.kernels.apply_phase_preservation(
                    projected_state,
                    recent_phase,
                    self.coherence_threshold
                )
            
            return projected_state
    
    def _project_to_subspace_cpu(self, state: torch.Tensor) -> torch.Tensor:
        """CPU implementation of subspace projection."""
        start, end = self.subspace_range
        mask = torch.zeros_like(state)
        mask[:, :, start:end] = 1.0
        return state * mask
        
    def _create_specialization_mask(self) -> torch.Tensor:
        """Create specialization mask for the instance."""
        mask = torch.zeros(self.state_node.quantum_state.size(-1))
        start, end = self.subspace_range
        mask[start:end] = 1.0
        return mask


class ClaudeNetwork:
    """Manages a network of specialized Claude instances."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_instances: int = 3,
        coherence_threshold: float = 0.7,
        phase_preservation: float = 0.5,
        cuda_manager: Optional['CUDAManager'] = None,
    ):
        """Initialize the Claude network with CUDA support."""
        self.hidden_size = hidden_size
        self.num_instances = num_instances
        self.coherence_threshold = coherence_threshold
        self.phase_preservation = phase_preservation
        
        # Initialize CUDA support
        self.cuda_manager = cuda_manager
        self.device = cuda_manager.device if cuda_manager else torch.device('cpu')
        if cuda_manager and cuda_manager.is_available():
            self.kernels = QRAFCUDAKernels()
            self.profiler = CUDAProfiler()
            self.use_cuda = True
        else:
            self.use_cuda = False
        
        # Initialize BitNet transformer for cross-instance communication
        self.transformer = BitNetTransformer(
            hidden_size=hidden_size,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512
        )
        
        # Initialize quantum state network with CUDA support
        self.state_network = QuantumStateNetwork(
            hidden_size=hidden_size,
            max_instances=num_instances,
            coherence_threshold=coherence_threshold,
            phase_preservation=phase_preservation,
            cuda_manager=cuda_manager
        )
        
        # Initialize specialized instances
        self.instances: Dict[int, SpecializedClaudeInstance] = {}
        self._initialize_instances()
        
        # Network metrics
        self.network_coherence = 1.0
        self.knowledge_transfer_efficiency = 1.0
        self.specialization_index = 1.0
        self.network_stability = 1.0
        
        # Initialize network state tensors on appropriate device
        if self.use_cuda:
            self._initialize_cuda_tensors()
            
    def _initialize_cuda_tensors(self):
        """Initialize CUDA-specific tensors for network operations."""
        with self.profiler.profile_operation("init_tensors", "cuda"):
            # Create network state tensors on GPU
            self.network_state = torch.zeros(
                self.num_instances,
                1,
                self.hidden_size,
                dtype=torch.complex64,
                device=self.device
            )
            
            # Initialize phase tensors for cross-instance communication
            self.phase_matrix = torch.zeros(
                self.num_instances,
                self.num_instances,
                dtype=torch.float32,
                device=self.device
            )
            
            # Create coherence tracking tensors
            self.coherence_matrix = torch.ones(
                self.num_instances,
                self.num_instances,
                dtype=torch.float32,
                device=self.device
            )
            
            # Initialize specialization masks for all instances
            self.specialization_masks = torch.zeros(
                self.num_instances,
                self.hidden_size,
                dtype=torch.float32,
                device=self.device
            )
            
    def _initialize_instances(self):
        """Initialize specialized Claude instances with enhanced state management and CUDA support."""
        specializations = [
            ("Abstract_Reasoning", (0, 255), 0.8),  # Higher coherence threshold
            ("Pattern_Recognition", (256, 511), 0.7),
            ("Causal_Analysis", (512, 767), 0.75)
        ]
        
        for idx, (spec_type, subspace_range, coherence_threshold) in enumerate(specializations):
            # Create Claude instance with specialized configuration and CUDA support
            claude = ClaudeV3_5Augmenter(
                hidden_size=self.hidden_size,
                quantum_config={
                    "adaptive_coherence": True,
                    "entanglement_threshold": coherence_threshold,
                    "phase_preservation": 0.8
                },
                cuda_manager=self.cuda_manager
            )
            
            # Initialize quantum state with specialization-specific phase
            phase = 2 * math.pi * idx / len(specializations)
            
            if self.use_cuda:
                with self.cuda_manager.error_context("init_instance"):
                    with self.cuda_manager.stream_context("quantum"):
                        initial_state = self._create_initial_state_cuda(
                            phase,
                            subspace_range
                        )
            else:
                initial_state = self._create_initial_state_cpu(
                    phase,
                    subspace_range
                )
            
            # Create state node with specialized initialization
            state_node = self.state_network.add_state(
                quantum_state=initial_state,
                temporal_index=0,
                coherence=1.0,
                specialization_mask=self._create_specialization_mask(subspace_range)
            )
            
            # Create specialized instance with CUDA support
            instance = SpecializedClaudeInstance(
                instance_id=idx,
                specialization=spec_type,
                claude=claude,
                state_node=state_node,
                subspace_range=subspace_range,
                coherence_threshold=coherence_threshold,
                cuda_manager=self.cuda_manager
            )
            
            self.instances[idx] = instance
            
            # Update CUDA tensors if available
            if self.use_cuda:
                self.network_state[idx] = initial_state
                self.specialization_masks[idx, subspace_range[0]:subspace_range[1]] = 1.0
                
    def _create_initial_state_cuda(
        self,
        phase: float,
        subspace_range: Tuple[int, int]
    ) -> torch.Tensor:
        """Create initial quantum state with CUDA optimization."""
        with self.profiler.profile_operation("create_state", "cuda"):
            # Create state tensor on GPU
            initial_state = torch.zeros(
                1, 1, self.hidden_size,
                dtype=torch.complex64,
                device=self.device
            )
            
            # Create phase tensor on GPU
            phase_tensor = torch.full(
                (1, 1, subspace_range[1] - subspace_range[0]),
                phase,
                dtype=torch.float32,
                device=self.device
            )
            
            # Apply phase with CUDA optimization
            initial_state[..., subspace_range[0]:subspace_range[1]] = \
                self.kernels.apply_phase_tensor(phase_tensor)
            
            # Apply quantization with CUDA
            initial_state = self.kernels.quantize_state(
                initial_state,
                self.transformer.get_quantization_params()
            )
            
            return initial_state
            
    def _create_initial_state_cpu(
        self,
        phase: float,
        subspace_range: Tuple[int, int]
    ) -> torch.Tensor:
        """CPU implementation of initial state creation."""
        initial_state = torch.zeros(1, 1, self.hidden_size, dtype=torch.complex64)
        phase_tensor = torch.full(
            (1, 1, subspace_range[1] - subspace_range[0]),
            phase,
            dtype=torch.float32
        )
        initial_state[..., subspace_range[0]:subspace_range[1]] = torch.exp(1j * phase_tensor)
        
        # Add quantization for initial state
        initial_state = self.transformer.quantize(initial_state)
        
        return initial_state
        
    def process_query(
        self,
        query: str,
        instance_id: Optional[int] = None
    ) -> Tuple[str, Dict[str, float]]:
        """Process query through appropriate Claude instance(s) with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("process_query"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._process_query_cuda(query, instance_id)
        else:
            return self._process_query_cpu(query, instance_id)
            
    def _process_query_cuda(
        self,
        query: str,
        instance_id: Optional[int] = None
    ) -> Tuple[str, Dict[str, float]]:
        """CUDA-optimized query processing."""
        with self.profiler.profile_operation("query_processing", "cuda"):
            if instance_id is not None:
                # Use specific instance
                if instance_id not in self.instances:
                    raise ValueError(f"Instance {instance_id} not found")
                    
                instance = self.instances[instance_id]
                
                # Process query with CUDA optimization
                response = instance.claude.process_query(query)
                
                # Update network state tensor
                response_state = instance.claude.get_embedding(response).to(self.device)
                self.network_state[instance_id] = response_state
                
                return response, self._get_instance_metrics(instance)
            
            # Use all instances with parallel processing
            responses = []
            response_states = []
            
            # Process query through all instances in parallel
            for instance in self.instances.values():
                # Project query to instance's subspace with CUDA
                instance_response = instance.claude.process_query(query)
                responses.append(instance_response)
                
                # Get response state and update network
                response_state = instance.claude.get_embedding(instance_response).to(self.device)
                response_states.append(response_state)
                self.network_state[instance.instance_id] = response_state
            
            # Stack response states for batch processing
            stacked_states = torch.stack(response_states)
            
            # Combine responses using quantum state combination
            combined_response = self._combine_responses_cuda(responses, stacked_states)
            
            return combined_response, self._get_network_metrics()
            
    def _process_query_cpu(
        self,
        query: str,
        instance_id: Optional[int] = None
    ) -> Tuple[str, Dict[str, float]]:
        """CPU implementation of query processing."""
        if instance_id is not None:
            # Use specific instance
            if instance_id not in self.instances:
                raise ValueError(f"Instance {instance_id} not found")
            instance = self.instances[instance_id]
            response = instance.claude.process_query(query)
            return response, self._get_instance_metrics(instance)
            
        # Use all instances and combine responses
        responses = []
        for instance in self.instances.values():
            # Project query to instance's subspace
            instance_response = instance.claude.process_query(query)
            responses.append(instance_response)
            
        # Combine responses using BitNet transformer
        combined_response = self._combine_responses(responses)
        
        return combined_response, self._get_network_metrics()
        
    def _combine_responses(
        self,
        responses: List[str]
    ) -> str:
        """Combine multiple responses using quantum state combination with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("combine_responses"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._combine_responses_cuda(responses)
        else:
            return self._combine_responses_cpu(responses)
            
    def _combine_responses_cuda(
        self,
        responses: List[str],
        response_states: Optional[torch.Tensor] = None
    ) -> str:
        """CUDA-optimized response combination."""
        with self.profiler.profile_operation("response_combination", "cuda"):
            if response_states is None:
                # Convert responses to quantum states if not provided
                response_states = []
                for response, instance in zip(responses, self.instances.values()):
                    embedding = instance.claude.get_embedding(response)
                    response_states.append(embedding)
                response_states = torch.stack(response_states).to(self.device)
            
            # Get batch dimensions
            batch_size, seq_length, hidden_size = response_states.size(1), response_states.size(2), response_states.size(3)
            
            # Reshape for batch processing
            reshaped_states = response_states.view(-1, seq_length, hidden_size)
            
            # Get coherence values for weighting
            coherence_values = []
            for instance in self.instances.values():
                coherence = instance.get_average_coherence()
                coherence_values.append(coherence)
            
            # Compute coherence-weighted combination with CUDA
            weights = torch.softmax(
                torch.tensor(coherence_values, device=self.device),
                dim=0
            )
            
            # Apply weighted combination with CUDA optimization
            combined_state = self.kernels.combine_quantum_states(
                reshaped_states,
                weights,
                self.coherence_threshold
            )
            
            # Apply quantization before transformer processing
            combined_state = self.kernels.quantize_state(
                combined_state,
                self.transformer.get_quantization_params()
            )
            
            # Process through transformer
            combined_state = self.transformer(combined_state)
            
            # Get phase histories for enhanced decoding
            phase_histories = []
            for instance in self.instances.values():
                if instance.phase_history:
                    phase_histories.append(instance.phase_history[-1])
            
            # Decode with phase preservation
            if phase_histories:
                # Combine phase histories with coherence weighting
                combined_phase = torch.tensor(phase_histories, device=self.device)
                weighted_phase = (combined_phase * weights).sum()
                
                # Apply phase-aware decoding
                return self.instances[0].claude.decode_embedding(
                    combined_state,
                    phase_history=[weighted_phase.cpu()]
                )
            else:
                return self.instances[0].claude.decode_embedding(combined_state)
            
    def _combine_responses_cpu(
        self,
        responses: List[str]
    ) -> str:
        """CPU implementation of response combination."""
        response_tensors = []
        phase_histories = []
        coherence_values = []
        
        # Collect response states and metadata
        for response, instance in zip(responses, self.instances.values()):
            embedding = instance.claude.get_embedding(response)
            response_tensors.append(embedding)
            
            if hasattr(instance.claude, 'phase_history'):
                phase_histories.append(instance.claude.phase_history)
            
            coherence = instance.claude.coherence_history[-1] if instance.claude.coherence_history else None
            coherence_values.append(coherence)
        
        # Stack and reshape responses
        stacked_responses = torch.stack(response_tensors)
        batch_size, seq_length, hidden_size = stacked_responses.size(1), stacked_responses.size(2), stacked_responses.size(3)
        reshaped_responses = stacked_responses.view(-1, seq_length, hidden_size)
        
        # Apply coherence-weighted combination
        if coherence_values:
            weights = torch.softmax(torch.tensor(coherence_values), dim=0)
            combined_tensor = torch.sum(
                reshaped_responses * weights.view(-1, 1, 1),
                dim=0,
                keepdim=True
            )
        else:
            combined_tensor = reshaped_responses.mean(dim=0, keepdim=True)
        
        # Add quantization before transformer processing
        combined_tensor = self.transformer.quantize(combined_tensor)
        combined_tensor = self.transformer(combined_tensor)
        
        # Decode with phase history if available
        instance = next(iter(self.instances.values()))
        if phase_histories:
            # Combine phase histories with coherence weighting
            combined_phase = torch.stack([
                history[-1] for history in phase_histories if history
            ]).mean(dim=0)
            return instance.claude.decode_embedding(
                combined_tensor,
                phase_history=[combined_phase]
            )
        else:
            return instance.claude.decode_embedding(combined_tensor)
        
    def _get_instance_metrics(
        self,
        instance: SpecializedClaudeInstance
    ) -> Dict[str, float]:
        """Get metrics for a specific instance with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("instance_metrics"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._get_instance_metrics_cuda(instance)
        else:
            return self._get_instance_metrics_cpu(instance)
            
    def _get_instance_metrics_cuda(
        self,
        instance: SpecializedClaudeInstance
    ) -> Dict[str, float]:
        """CUDA-optimized instance metrics computation."""
        with self.profiler.profile_operation("instance_metrics", "cuda"):
            # Get coherence with existing CUDA optimization
            coherence = instance.get_average_coherence()
            
            # Compute phase mean with CUDA
            if instance.phase_history:
                phase_tensor = torch.tensor(
                    instance.phase_history,
                    device=self.device
                )
                phase = phase_tensor.mean().item()
            else:
                phase = 0.0
                
            # Get specialization score with CUDA optimization
            specialization_score = self._compute_specialization_score(instance)
            
            return {
                'coherence': coherence,
                'phase': phase,
                'specialization_score': specialization_score
            }
            
    def _get_instance_metrics_cpu(
        self,
        instance: SpecializedClaudeInstance
    ) -> Dict[str, float]:
        """CPU implementation of instance metrics computation."""
        return {
            'coherence': instance.get_average_coherence(),
            'phase': np.mean(instance.phase_history) if instance.phase_history else 0.0,
            'specialization_score': self._compute_specialization_score(instance)
        }
        
    def _compute_specialization_score(
        self,
        instance: SpecializedClaudeInstance
    ) -> float:
        """Compute how well an instance maintains its specialization with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("specialization_score"):
                with self.cuda_manager.stream_context("quantum"):
                    return self._compute_specialization_score_cuda(instance)
        else:
            return self._compute_specialization_score_cpu(instance)
            
    def _compute_specialization_score_cuda(
        self,
        instance: SpecializedClaudeInstance
    ) -> float:
        """CUDA-optimized specialization score computation."""
        with self.profiler.profile_operation("specialization_score", "cuda"):
            # Move state to GPU if needed
            state = instance.state_node.quantum_state
            if not state.is_cuda:
                state = state.to(self.device)
                
            # Project state with existing CUDA optimization
            projected = instance.project_to_subspace(state)
            
            # Compute norms with CUDA efficiency
            state_norm = torch.norm(state)
            projected_norm = torch.norm(projected)
            
            # Handle zero norm case
            if state_norm == 0:
                return 0.0
                
            return (projected_norm / state_norm).item()
            
    def _compute_specialization_score_cpu(
        self,
        instance: SpecializedClaudeInstance
    ) -> float:
        """CPU implementation of specialization score computation."""
        state = instance.state_node.quantum_state
        projected = instance.project_to_subspace(state)
        return torch.norm(projected) / torch.norm(state)
        
    def synchronize_phases(self):
        """Synchronize phases across all instances with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("phase_sync"):
                with self.cuda_manager.stream_context("quantum"):
                    self._synchronize_phases_cuda()
        else:
            self._synchronize_phases_cpu()
            
    def _synchronize_phases_cuda(self):
        """CUDA-optimized phase synchronization."""
        with self.profiler.profile_operation("phase_sync", "cuda"):
            # Collect all quantum states in batch
            states = torch.stack([
                instance.state_node.quantum_state.to(self.device)
                for instance in self.instances.values()
            ])
            
            # Compute phases in parallel
            phases = self.kernels.compute_quantum_phases(states)
            mean_phase = phases.mean()
            
            # Compute phase differences and corrections
            phase_diffs = mean_phase - phases
            
            # Get coherence values for correction modulation
            coherence_values = torch.tensor([
                instance.get_average_coherence()
                for instance in self.instances.values()
            ], device=self.device)
            
            # Compute correction strengths with CUDA
            correction_strengths = torch.sigmoid(coherence_values * 5.0)
            
            # Apply corrections in parallel
            corrected_states = self.kernels.apply_phase_corrections(
                states,
                phase_diffs,
                correction_strengths
            )
            
            # Update instance states and metrics
            for idx, instance in enumerate(self.instances.values()):
                instance.state_node.quantum_state = corrected_states[idx]
                instance.update_metrics(
                    phase=phases[idx].item(),
                    coherence=coherence_values[idx].item()
                )
                
    def _synchronize_phases_cpu(self):
        """CPU implementation of phase synchronization."""
        # Get all quantum states
        states = [
            instance.state_node.quantum_state 
            for instance in self.instances.values()
        ]
        phases = [
            torch.angle(state).mean()
            for state in states
        ]
        
        # Compute mean phase
        mean_phase = torch.mean(torch.stack(phases))
        
        # Apply phase corrections
        for instance in self.instances.values():
            state = instance.state_node.quantum_state
            current_phase = torch.angle(state).mean()
            phase_diff = mean_phase - current_phase
            
            # Apply correction with coherence-based modulation
            coherence = instance.get_average_coherence()
            correction_strength = torch.sigmoid(torch.tensor(coherence * 5.0))
            
            corrected_state = state * torch.exp(1j * phase_diff * correction_strength)
            instance.state_node.quantum_state = corrected_state
            
            # Update metrics
            instance.update_metrics(
                phase=current_phase.item(),
                coherence=coherence
            )
            
    def update_network_metrics(self):
        """Update network-wide metrics with CUDA optimization."""
        if self.use_cuda:
            with self.cuda_manager.error_context("network_metrics"):
                with self.cuda_manager.stream_context("quantum"):
                    self._update_network_metrics_cuda()
        else:
            self._update_network_metrics_cpu()
            
    def _update_network_metrics_cuda(self):
        """CUDA-optimized network metrics update."""
        with self.profiler.profile_operation("metrics_update", "cuda"):
            # Get coherence values in batch
            coherence_values = torch.tensor([
                instance.get_average_coherence()
                for instance in self.instances.values()
            ], device=self.device)
            self.network_coherence = coherence_values.mean().item()
            
            # Collect quantum states in batch
            states = torch.stack([
                instance.state_node.quantum_state.to(self.device)
                for instance in self.instances.values()
            ])
            
            # Compute knowledge transfer efficiency with CUDA
            similarities = self.kernels.compute_state_similarities(
                states,
                self.coherence_threshold
            )
            self.knowledge_transfer_efficiency = similarities.mean().item()
            
            # Compute specialization scores in parallel
            specialization_scores = torch.zeros(
                len(self.instances),
                device=self.device
            )
            for idx, instance in enumerate(self.instances.values()):
                projected = instance.project_to_subspace(states[idx])
                specialization_scores[idx] = (
                    torch.norm(projected) / torch.norm(states[idx])
                )
            self.specialization_index = specialization_scores.mean().item()
            
            # Update network stability with phase history analysis
            phase_stds = torch.tensor([
                torch.std(torch.tensor(instance.phase_history[-100:], device=self.device))
                if len(instance.phase_history) > 0 else torch.tensor(0.0, device=self.device)
                for instance in self.instances.values()
            ])
            self.network_stability = 1.0 / (1.0 + phase_stds.mean().item())
            
    def _update_network_metrics_cpu(self):
        """CPU implementation of network metrics update."""
        # Update network coherence
        coherences = [
            instance.get_average_coherence()
            for instance in self.instances.values()
        ]
        self.network_coherence = np.mean(coherences) if coherences else 0.0
        
        # Update knowledge transfer efficiency
        states = [
            instance.state_node.quantum_state
            for instance in self.instances.values()
        ]
        similarities = []
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i < j:
                    norm1 = torch.norm(state1)
                    norm2 = torch.norm(state2)
                    if norm1 > 0 and norm2 > 0:
                        sim = torch.abs(
                            torch.sum(state1 * state2.conj())
                        ).item() / (norm1 * norm2).item()
                        similarities.append(sim)
        self.knowledge_transfer_efficiency = np.mean(similarities) if similarities else 0.0
        
        # Update specialization index
        specialization_scores = [
            self._compute_specialization_score(instance)
            for instance in self.instances.values()
        ]
        self.specialization_index = np.mean(specialization_scores) if specialization_scores else 0.0
        
        # Update network stability
        phase_stds = [
            np.std(instance.phase_history) if instance.phase_history else 0.0
            for instance in self.instances.values()
        ]
        self.network_stability = 1.0 / (1.0 + np.mean(phase_stds)) if phase_stds else 1.0 

    def _get_network_metrics(self) -> Dict[str, float]:
        """Get network-wide metrics."""
        return {
            'network_coherence': self.network_coherence,
            'knowledge_transfer_efficiency': self.knowledge_transfer_efficiency,
            'specialization_index': self.specialization_index,
            'network_stability': self.network_stability
        } 