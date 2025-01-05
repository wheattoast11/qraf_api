from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import torch
from pydantic import BaseModel, Field
import uuid
from ..core.quantum_state_manager import QuantumStateManager, StateConfig

router = APIRouter(prefix="/state", tags=["state"])

class StateRequest(BaseModel):
    """Request model for state operations"""
    tensor_data: List[float] = Field(..., description="Flattened tensor data")
    shape: List[int] = Field(..., description="Tensor shape")
    optimize: bool = Field(True, description="Whether to apply optimizations")

class StateResponse(BaseModel):
    """Response model for state operations"""
    state_id: str
    coherence: float
    is_optimized: bool

# Dependency for state manager
async def get_state_manager():
    # In production, this would be properly initialized and managed
    config = StateConfig()
    return QuantumStateManager(config)

@router.post("/process")
async def process_state(
    request: StateRequest,
    state_manager: QuantumStateManager = Depends(get_state_manager)
) -> StateResponse:
    """Process a quantum state"""
    try:
        # Convert request data to tensor
        tensor_data = torch.tensor(request.tensor_data)
        tensor = tensor_data.reshape(request.shape)
        
        # Generate unique ID for this state
        state_id = str(uuid.uuid4())
        
        # Process the state
        processed_state = await state_manager.process_state(
            state_id=state_id,
            input_state=tensor,
            optimize=request.optimize
        )
        
        # Calculate coherence
        coherence = state_manager._calculate_coherence(processed_state)
        
        return StateResponse(
            state_id=state_id,
            coherence=coherence,
            is_optimized=request.optimize
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retrieve/{state_id}")
async def retrieve_state(
    state_id: str,
    state_manager: QuantumStateManager = Depends(get_state_manager)
) -> Dict:
    """Retrieve a processed state"""
    state = await state_manager.retrieve_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="State not found")
    
    return {
        "state_id": state_id,
        "tensor_data": state.flatten().tolist(),
        "shape": list(state.shape),
        "coherence": state_manager._calculate_coherence(state)
    }

@router.post("/attention")
async def apply_attention(
    query_state_id: str,
    key_state_ids: List[str],
    state_manager: QuantumStateManager = Depends(get_state_manager)
) -> Dict:
    """Apply attention across states"""
    # Retrieve states
    query_state = await state_manager.retrieve_state(query_state_id)
    if query_state is None:
        raise HTTPException(status_code=404, detail="Query state not found")
    
    key_states = []
    for key_id in key_state_ids:
        state = await state_manager.retrieve_state(key_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Key state {key_id} not found")
        key_states.append(state)
    
    # Apply attention
    result = await state_manager.apply_attention(query_state, key_states)
    
    # Generate new state ID
    new_state_id = str(uuid.uuid4())
    await state_manager.process_state(new_state_id, result)
    
    return {
        "new_state_id": new_state_id,
        "tensor_data": result.flatten().tolist(),
        "shape": list(result.shape),
        "coherence": state_manager._calculate_coherence(result)
    }

@router.delete("/{state_id}")
async def cleanup_state(
    state_id: str,
    state_manager: QuantumStateManager = Depends(get_state_manager)
):
    """Clean up a state's resources"""
    await state_manager.cleanup(state_id)
    return {"status": "success", "message": f"State {state_id} cleaned up"} 