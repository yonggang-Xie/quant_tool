"""
Int8 Quantization Tool for PyTorch

A comprehensive toolkit for implementing int8 quantization in PyTorch models.
Supports both symmetric and asymmetric quantization schemes with per-tensor
and per-channel quantization options.

Main Components:
- Int8Quantizer: Core quantization class
- QuantizedLinear, QuantizedConv2d: Quantized layer implementations
- Utility functions for analysis and benchmarking
- Sample models for testing

Example usage:
    from quant_tool import Int8Quantizer
    
    # Create quantizer
    quantizer = Int8Quantizer(symmetric=True, per_channel=False)
    
    # Calibrate and quantize tensor
    quantizer.calibrate(tensor)
    quantized = quantizer.quantize(tensor)
    dequantized = quantizer.dequantize(quantized)
"""

from .quantizer import (
    Int8Quantizer,
    QuantizedLinear,
    QuantizedConv2d,
    quantize_model
)

from .utils import (
    calculate_model_size,
    benchmark_inference,
    calculate_quantization_error,
    analyze_tensor_distribution,
    compare_models,
    print_quantization_summary
)

from .models import (
    SimpleMLPClassifier,
    SimpleCNN,
    SimpleResNet,
    SimpleTransformer,
    ModelFactory,
    create_sample_models,
    generate_sample_data
)

__version__ = "1.0.0"
__author__ = "Quantization Tool Development Team"
__email__ = "your.email@example.com"

__all__ = [
    # Core quantization
    'Int8Quantizer',
    'QuantizedLinear', 
    'QuantizedConv2d',
    'quantize_model',
    
    # Utilities
    'calculate_model_size',
    'benchmark_inference',
    'calculate_quantization_error',
    'analyze_tensor_distribution',
    'compare_models',
    'print_quantization_summary',
    
    # Models
    'SimpleMLPClassifier',
    'SimpleCNN',
    'SimpleResNet',
    'SimpleTransformer',
    'ModelFactory',
    'create_sample_models',
    'generate_sample_data'
]
