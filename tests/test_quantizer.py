#!/usr/bin/env python3
"""
Unit tests for the Int8 quantization tool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from quantizer import Int8Quantizer, QuantizedLinear, QuantizedConv2d
from utils import calculate_quantization_error, calculate_model_size
from models import SimpleMLPClassifier


class TestInt8Quantizer:
    """Test cases for Int8Quantizer class."""
    
    def test_symmetric_per_tensor_quantization(self):
        """Test symmetric per-tensor quantization."""
        quantizer = Int8Quantizer(symmetric=True, per_channel=False)
        tensor = torch.randn(50, 20) * 2.0
        
        # Calibrate and quantize
        quantizer.calibrate(tensor)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Check quantized range
        assert quantized.min().item() >= -127
        assert quantized.max().item() <= 127
        assert quantized.dtype == torch.int8
        
        # Check dequantized shape
        assert dequantized.shape == tensor.shape
        assert dequantized.dtype == torch.float32
        
        # Check zero-point is zero for symmetric
        params = quantizer.get_quantization_params()
        assert params['zero_point'] == 0
        assert params['symmetric'] == True
    
    def test_asymmetric_per_tensor_quantization(self):
        """Test asymmetric per-tensor quantization."""
        quantizer = Int8Quantizer(symmetric=False, per_channel=False)
        tensor = torch.randn(50, 20) * 2.0 + 1.0  # Add offset
        
        # Calibrate and quantize
        quantizer.calibrate(tensor)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Check quantized range
        assert quantized.min().item() >= -128
        assert quantized.max().item() <= 127
        assert quantized.dtype == torch.int8
        
        # Check parameters
        params = quantizer.get_quantization_params()
        assert params['symmetric'] == False
        assert isinstance(params['zero_point'], int)
    
    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        quantizer = Int8Quantizer(symmetric=True, per_channel=True, channel_axis=0)
        tensor = torch.randn(10, 20)
        
        # Make channels have different scales
        for i in range(10):
            tensor[i] *= (i + 1) * 0.5
        
        # Calibrate and quantize
        quantizer.calibrate(tensor)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Check parameters are per-channel
        params = quantizer.get_quantization_params()
        assert isinstance(params['scale'], torch.Tensor)
        assert len(params['scale']) == 10  # Number of channels
        
        # Check shapes
        assert quantized.shape == tensor.shape
        assert dequantized.shape == tensor.shape
    
    def test_quantization_error_bounds(self):
        """Test that quantization error is within reasonable bounds."""
        quantizer = Int8Quantizer(symmetric=True, per_channel=False)
        tensor = torch.randn(100, 50)
        
        quantizer.calibrate(tensor)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Calculate error metrics
        error_metrics = calculate_quantization_error(tensor, quantized, dequantized)
        
        # Check that cosine similarity is high (> 0.9)
        assert error_metrics['cosine_similarity'] > 0.9
        
        # Check that relative error is reasonable (< 0.1)
        assert error_metrics['relative_error'] < 0.1
    
    def test_calibration_consistency(self):
        """Test that calibration produces consistent results."""
        tensor = torch.randn(100, 50)
        
        # Create two identical quantizers
        quantizer1 = Int8Quantizer(symmetric=True, per_channel=False)
        quantizer2 = Int8Quantizer(symmetric=True, per_channel=False)
        
        # Calibrate both with same data
        quantizer1.calibrate(tensor)
        quantizer2.calibrate(tensor)
        
        # Check parameters are identical
        params1 = quantizer1.get_quantization_params()
        params2 = quantizer2.get_quantization_params()
        
        assert abs(params1['scale'] - params2['scale']) < 1e-6
        assert params1['zero_point'] == params2['zero_point']
    
    def test_edge_cases(self):
        """Test edge cases like zero tensors, single values, etc."""
        quantizer = Int8Quantizer(symmetric=True, per_channel=False)
        
        # Test zero tensor
        zero_tensor = torch.zeros(10, 10)
        quantizer.calibrate(zero_tensor)
        quantized = quantizer.quantize(zero_tensor)
        dequantized = quantizer.dequantize(quantized)
        
        assert torch.allclose(quantized, torch.zeros_like(quantized))
        assert torch.allclose(dequantized, torch.zeros_like(dequantized))
        
        # Test single value tensor
        single_tensor = torch.ones(10, 10) * 5.0
        quantizer.calibrate(single_tensor)
        quantized = quantizer.quantize(single_tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Should be close to original
        assert torch.allclose(dequantized, single_tensor, rtol=0.1)


class TestQuantizedLayers:
    """Test cases for quantized layer implementations."""
    
    def test_quantized_linear_layer(self):
        """Test QuantizedLinear layer."""
        layer = QuantizedLinear(10, 5, bias=True)
        
        # Test forward pass before quantization
        input_tensor = torch.randn(32, 10)
        output = layer(input_tensor)
        assert output.shape == (32, 5)
        
        # Quantize weights
        layer.quantize_weights()
        assert layer.is_quantized == True
        assert layer.quantized_weight is not None
        
        # Test forward pass after quantization
        output_quantized = layer(input_tensor)
        assert output_quantized.shape == (32, 5)
        
        # Outputs should be similar but not identical
        mse = torch.mean((output - output_quantized) ** 2).item()
        assert mse < 1.0  # Reasonable error bound
    
    def test_quantized_conv2d_layer(self):
        """Test QuantizedConv2d layer."""
        layer = QuantizedConv2d(3, 16, kernel_size=3, padding=1, bias=True)
        
        # Test forward pass before quantization
        input_tensor = torch.randn(8, 3, 32, 32)
        output = layer(input_tensor)
        assert output.shape == (8, 16, 32, 32)
        
        # Quantize weights
        layer.quantize_weights()
        assert layer.is_quantized == True
        
        # Test forward pass after quantization
        output_quantized = layer(input_tensor)
        assert output_quantized.shape == (8, 16, 32, 32)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_calculate_model_size(self):
        """Test model size calculation."""
        model = SimpleMLPClassifier(100, [50, 25], 10)
        size_info = calculate_model_size(model)
        
        assert 'total_mb' in size_info
        assert 'parameters_mb' in size_info
        assert 'buffers_mb' in size_info
        assert size_info['total_mb'] > 0
    
    def test_quantization_error_calculation(self):
        """Test quantization error calculation."""
        original = torch.randn(100, 50)
        quantized = torch.randint(-128, 127, (100, 50), dtype=torch.int8)
        dequantized = quantized.float() * 0.1  # Simple dequantization
        
        error_metrics = calculate_quantization_error(original, quantized, dequantized)
        
        required_keys = ['mse', 'mae', 'snr_db', 'psnr_db', 'cosine_similarity', 'relative_error']
        for key in required_keys:
            assert key in error_metrics
            assert isinstance(error_metrics[key], float)


def test_integration():
    """Integration test combining multiple components."""
    # Create model
    model = SimpleMLPClassifier(50, [32, 16], 5)
    
    # Create sample data
    input_data = torch.randn(10, 50)
    
    # Get original output
    model.eval()
    with torch.no_grad():
        original_output = model(input_data)
    
    # Test quantization of model weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            quantizer = Int8Quantizer(symmetric=True, per_channel=False)
            
            # Quantize weights
            original_weight = module.weight.data.clone()
            quantizer.calibrate(original_weight)
            quantized_weight = quantizer.quantize(original_weight)
            dequantized_weight = quantizer.dequantize(quantized_weight)
            
            # Replace weights temporarily
            module.weight.data = dequantized_weight
    
    # Get quantized output
    with torch.no_grad():
        quantized_output = model(input_data)
    
    # Check that outputs are similar
    mse = torch.mean((original_output - quantized_output) ** 2).item()
    assert mse < 1.0  # Reasonable error bound


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_suite = [
        TestInt8Quantizer(),
        TestQuantizedLayers(),
        TestUtilityFunctions()
    ]
    
    print("Running Int8 Quantization Tool Tests...")
    
    for test_class in test_suite:
        class_name = test_class.__class__.__name__
        print(f"\n--- {class_name} ---")
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                print(f"Running {method_name}...", end=" ")
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print("PASSED")
                except Exception as e:
                    print(f"FAILED: {e}")
    
    # Run integration test
    print(f"\n--- Integration Test ---")
    print("Running test_integration...", end=" ")
    try:
        test_integration()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\nTest suite completed!")
