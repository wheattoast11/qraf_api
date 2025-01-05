from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import torch
from pydantic import BaseModel, Field
from ..core.quantum_state_manager import QuantumStateManager, StateConfig
from qraf.utils.density_optimization import DensityOptimizer
from qraf.utils.quantization import QuantizationManager

router = APIRouter(prefix="/optimize", tags=["optimization"])

class OptimizationRequest(BaseModel):
    """Request model for optimization operations"""
    state_id: str = Field(..., description="ID of state to optimize")
    optimization_type: str = Field(..., description="Type of optimization to apply", 
                                 enum=["density", "quantization", "both"])
    parameters: Dict[str, float] = Field(default_factory=dict, 
                                       description="Optional optimization parameters")

class OptimizationResponse(BaseModel):
    """Response model for optimization operations"""
    new_state_id: str
    original_coherence: float
    optimized_coherence: float
    optimization_gain: float
    optimization_type: str

# Dependency for optimizers
async def get_optimizers():
    return {
        "density": DensityOptimizer(),
        "quantization": QuantizationManager()
    }

@router.post("/state")
async def optimize_state(
    request: OptimizationRequest,
    state_manager: QuantumStateManager = Depends(get_state_manager),
    optimizers: Dict = Depends(get_optimizers)
) -> OptimizationResponse:
    """Optimize a quantum state"""
    try:
        # Retrieve original state
        original_state = await state_manager.retrieve_state(request.state_id)
        if original_state is None:
            raise HTTPException(status_code=404, detail="State not found")
            
        # Calculate original coherence
        original_coherence = state_manager._calculate_coherence(original_state)
        
        # Apply optimizations
        state = original_state
        if request.optimization_type in ["density", "both"]:
            state = optimizers["density"].optimize(
                state,
                **request.parameters.get("density", {})
            )
            
        if request.optimization_type in ["quantization", "both"]:
            state = optimizers["quantization"].quantize(
                state,
                **request.parameters.get("quantization", {})
            )
            
        # Store optimized state
        new_state_id = str(uuid.uuid4())
        await state_manager.process_state(
            new_state_id,
            state,
            optimize=False  # Already optimized
        )
        
        # Calculate new coherence
        optimized_coherence = state_manager._calculate_coherence(state)
        
        return OptimizationResponse(
            new_state_id=new_state_id,
            original_coherence=original_coherence,
            optimized_coherence=optimized_coherence,
            optimization_gain=optimized_coherence - original_coherence,
            optimization_type=request.optimization_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def optimize_batch(
    state_ids: List[str],
    optimization_type: str,
    state_manager: QuantumStateManager = Depends(get_state_manager),
    optimizers: Dict = Depends(get_optimizers)
) -> Dict[str, OptimizationResponse]:
    """Optimize multiple states in parallel"""
    results = {}
    for state_id in state_ids:
        request = OptimizationRequest(
            state_id=state_id,
            optimization_type=optimization_type
        )
        try:
            result = await optimize_state(request, state_manager, optimizers)
            results[state_id] = result
        except HTTPException as e:
            results[state_id] = {"error": e.detail}
            
    return results

@router.get("/parameters")
async def get_optimization_parameters() -> Dict[str, Dict]:
    """Get available optimization parameters and their defaults"""
    return {
        "density": {
            "eigenvalue_threshold": 0.1,
            "preserve_phase": True,
            "max_iterations": 100
        },
        "quantization": {
            "bits": 8,
            "scheme": "linear",
            "preserve_sparsity": True
        }
    } 