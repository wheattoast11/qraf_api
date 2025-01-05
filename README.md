# QRAF: Quantum Reasoning Augmentation Framework

QRAF is a cutting-edge framework that combines quantum computing concepts with AI reasoning capabilities to enhance symbolic and mathematical proof generation. At its core, it uses a BitNet transformer architecture with 1.58-bit quantization for efficient state representation and processing.

## Features

### Core QRAF Features
- **Quantum Proof Search**: Advanced proof search using quantum-inspired algorithms
- **Density Optimization**: Information density tracking and optimization
- **Claude Integration**: Seamless integration with Claude's reasoning capabilities

### BitNet Architecture Features
- **Quantum-Inspired Architecture**: Leverages quantum computing principles for enhanced state representation and processing
- **1.58-bit Quantization**: Efficient binary-like quantization for reduced memory footprint and faster computation
- **State Preservation**: Maintains quantum state coherence through phase-aware processing
- **Flexible Configuration**: Highly configurable architecture through BitNetConfig
- **Modular Design**: Separate components for attention, transformer, and decoder
- **Tokenizer Integration**: Built-in tokenizer with GPT-2 compatibility
- **Save/Load Support**: Easy model persistence and loading

## Installation

1. Install Python requirements:
```bash
pip install -r requirements.txt
```

2. Install CUTLASS from source:
```bash
# Clone CUTLASS repository
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Configure with CMake
mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS=86 # Adjust for your GPU architecture
cmake --build . --config Release

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/cutlass/build/tools/library/python
```

Note: Replace the CUDA architecture flag (`86` for RTX 30 series) with your GPU's architecture:
- 70 for Volta (V100)
- 75 for Turing (RTX 20 series)
- 86 for Ampere (RTX 30 series)
- 89 for Ada Lovelace (RTX 40 series)

## Usage

### Basic Example

```python
import torch
from qraf.core.bitnet_transformer import BitNetConfig, BitNetDecoder
from qraf.core.proof_search import QuantumProofPathfinder
from qraf.interfaces.claude_augmenter import ClaudeReasoningAugmenter

# Initialize components
proof_finder = QuantumProofPathfinder()
augmenter = ClaudeReasoningAugmenter()

# Example: Generate proof for a theorem
import sympy as sp
x, y = sp.symbols('x y')
theorem = sp.Eq(x**2 + y**2, (x + y)**2)
proof = proof_finder.generate_proof_report(theorem)

# Initialize BitNet configuration
config = BitNetConfig(
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    is_decoder=True
)

# Create decoder and process quantum states
decoder = BitNetDecoder(config)
quantum_state = torch.randn(1, 10, config.hidden_size)
output = decoder.decode(
    quantum_state=quantum_state,
    use_sampling=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
```

### Advanced Configuration

```python
config = BitNetConfig(
    # Model architecture
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    vocab_size=50257,
    
    # Dropout rates
    attention_dropout=0.1,
    hidden_dropout=0.1,
    
    # Decoder settings
    is_decoder=True,
    max_length=128,
    min_length=0,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    early_stopping=False,
    
    # Quantization
    quantization_bits=2,  # 1.58-bit quantization rounded up
    quantization_method="stochastic",  # or "deterministic"
    
    # Quantum-inspired settings
    phase_preservation=True,
    coherence_threshold=0.7,
    interference_threshold=0.9,
    entanglement_preservation=True
)
```

## Architecture

### QRAF Components
The framework consists of several integrated components:
1. **Proof Search Engine**: Quantum-inspired pathfinding for mathematical proofs
2. **Density Optimizer**: Manages information density and coherence
3. **Claude Interface**: Connects with Claude for enhanced reasoning

### BitNet Components
The BitNet implementation provides the quantum processing backbone:
1. **BitNetConfig**: Configuration class for model architecture and hyperparameters
2. **BitNetTokenizer**: Tokenizer with quantum state preservation capabilities
3. **BitNetAttention**: Quantum-inspired attention mechanism
4. **BitNetTransformer**: Core transformer architecture with quantization
5. **BitNetDecoder**: Text generation decoder with state management

### Quantization Details
The implementation uses a novel 1.58-bit quantization scheme:
- **Deterministic Mode**: Simple rounding to nearest quantization level
- **Stochastic Mode**: Noise-based probabilistic quantization
- **Level Range**: [-1, 1] with 2^bits discrete levels

### Quantum-Inspired Features
- **Phase Preservation**: Maintains quantum phase information during processing
- **Coherence Management**: Tracks and optimizes state coherence
- **Interference Patterns**: Utilizes quantum interference for enhanced processing
- **Entanglement**: Preserves quantum entanglement patterns

## Project Structure

```
qraf/
├── src/
│   └── qraf/
│       ├── core/
│       │   ├── bitnet_transformer.py
│       │   └── proof_search.py
│       ├── utils/
│       │   ├── density_optimization.py
│       │   └── quantization.py
│       └── interfaces/
│           └── claude_augmenter.py
├── tests/
├── notebooks/
├── requirements.txt
└── setup.py
```

## Development

```bash
# Run tests
pytest

# Format code
black src tests
isort src tests

# Type checking
mypy src
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use QRAF in your research, please cite:

```bibtex
@software{qraf2024,
    title={QRAF: Quantum Reasoning Augmentation Framework},
    author={QRAF Team},
    year={2024},
    url={https://github.com/qraf-team/qraf}
}

@misc{bitnet2024,
    title={BitNet: Quantum-Inspired Transformer with 1.58-bit Quantization},
    author={QRAF Team},
    year={2024},
    publisher={GitHub},
    howpublished={\url{https://github.com/qraf-team/qraf}}
}
``` 