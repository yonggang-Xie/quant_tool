# Int8 Quantization Tool for PyTorch

A comprehensive toolkit for implementing int8 quantization in PyTorch models without using off-the-shelf quantization libraries. This tool provides custom implementations of quantization algorithms, quantized layers, and analysis utilities.

## Features

- **Custom Int8 Quantization**: Implementation of symmetric and asymmetric quantization schemes
- **Per-tensor and Per-channel Quantization**: Support for both quantization granularities
- **Quantized Layers**: Custom implementations of QuantizedLinear and QuantizedConv2d layers
- **Model Analysis**: Comprehensive tools for analyzing quantization effects
- **Performance Benchmarking**: Tools to measure model size reduction and inference speedup
- **Sample Models**: Various neural network architectures for testing
- **Visualization**: Tools to visualize quantization effects (optional with matplotlib)

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9.0+
- NumPy 1.20.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yonggang-Xie/quant_tool.git
cd quant_tool
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Tensor Quantization

```python
import torch
from quant_tool import Int8Quantizer

# Create a sample tensor
tensor = torch.randn(100, 50) * 2.0

# Create quantizer (symmetric, per-tensor)
quantizer = Int8Quantizer(symmetric=True, per_channel=False)

# Calibrate quantizer
quantizer.calibrate(tensor)

# Quantize tensor
quantized = quantizer.quantize(tensor)
print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")

# Dequantize back to float32
dequantized = quantizer.dequantize(quantized)

# Calculate quantization error
mse = torch.mean((tensor - dequantized) ** 2)
print(f"Quantization MSE: {mse:.8f}")
```

### Model Quantization

```python
import torch
from quant_tool import SimpleMLPClassifier, QuantizedLinear
from quant_tool.utils import calculate_model_size, benchmark_inference

# Create original model
model = SimpleMLPClassifier(784, [256, 128], 10)

# Generate sample data for calibration
calibration_data = torch.randn(100, 784)

# Replace layers with quantized versions (simplified example)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Create quantized layer
        quant_layer = QuantizedLinear(module.in_features, module.out_features)
        quant_layer.weight.data = module.weight.data.clone()
        if module.bias is not None:
            quant_layer.bias.data = module.bias.data.clone()
        quant_layer.quantize_weights()
        
        # Replace in model (requires proper model traversal in practice)
        # setattr(parent_module, name, quant_layer)

# Analyze model size reduction
original_size = calculate_model_size(model)
print(f"Original model size: {original_size['total_mb']:.2f} MB")
```

## Quantization Schemes

### Symmetric Quantization
Maps float32 values to the range [-127, 127] using a single scale parameter:
```
quantized = round(tensor / scale)
```

### Asymmetric Quantization  
Maps float32 values to the full int8 range [-128, 127] using scale and zero-point:
```
quantized = round(tensor / scale + zero_point)
```

### Per-tensor vs Per-channel
- **Per-tensor**: Single scale/zero-point for entire tensor
- **Per-channel**: Separate scale/zero-point for each channel (better accuracy)

## Examples

Run the provided examples to see the quantization tool in action:

```bash
# Basic tensor quantization examples
python examples/basic_quantization.py

# Model quantization and comparison
python examples/model_comparison.py
```

## API Reference

### Core Classes

#### `Int8Quantizer`
Main quantization class supporting various quantization schemes.

**Parameters:**
- `symmetric` (bool): Use symmetric quantization (default: True)
- `per_channel` (bool): Use per-channel quantization (default: False)  
- `channel_axis` (int): Axis for per-channel quantization (default: 0)

**Methods:**
- `calibrate(tensor)`: Calibrate quantizer with sample data
- `quantize(tensor)`: Quantize float32 tensor to int8
- `dequantize(tensor)`: Dequantize int8 tensor to float32
- `get_quantization_params()`: Get scale and zero-point parameters

#### `QuantizedLinear`
Quantized linear layer implementation.

#### `QuantizedConv2d`
Quantized 2D convolution layer implementation.

### Utility Functions

- `calculate_model_size(model)`: Calculate model memory footprint
- `benchmark_inference(model, input_tensor)`: Benchmark inference time
- `calculate_quantization_error(original, quantized, dequantized)`: Calculate error metrics
- `compare_models(original_model, quantized_model, test_input)`: Comprehensive model comparison
- `print_quantization_summary(quantizer, tensor)`: Print detailed quantization analysis

## Project Structure

```
quant_tool/
├── __init__.py              # Package initialization
├── quantizer.py             # Core quantization implementation
├── utils.py                 # Utility functions
├── models.py                # Sample neural network models
├── examples/
│   ├── basic_quantization.py    # Basic quantization examples
│   └── model_comparison.py      # Model quantization examples
├── tests/
│   └── test_quantizer.py        # Unit tests
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## Testing

Run the unit tests to verify the implementation:

```bash
python -m pytest tests/ -v
```

## Performance Considerations

- **Memory Reduction**: Int8 quantization typically reduces model size by ~4x
- **Inference Speed**: Speedup depends on hardware support for int8 operations
- **Accuracy Trade-off**: Some accuracy loss is expected; use calibration data representative of your use case
- **Layer Sensitivity**: Different layers have varying sensitivity to quantization

## Limitations

- This is an educational/research implementation
- Production deployments should consider hardware-optimized quantization libraries
- Limited to int8 quantization (no int4, int16, etc.)
- Quantized operations use dequantize-compute-quantize pattern (not true int8 arithmetic)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{int8_quantization_tool,
  title={Int8 Quantization Tool for PyTorch},
  author={Quantization Tool Development Team},
  year={2025},
  url={https://github.com/yonggang-Xie/quant_tool}
}
```

## References

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
