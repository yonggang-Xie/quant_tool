#!/usr/bin/env python3
"""
Basic quantization example demonstrating tensor-level int8 quantization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from quantizer import Int8Quantizer
from utils import print_quantization_summary, visualize_quantization_effects


def demonstrate_tensor_quantization():
    """Demonstrate basic tensor quantization with different schemes."""
    
    print("=" * 60)
    print("BASIC INT8 TENSOR QUANTIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample tensors with different distributions
    tensors = {
        'normal': torch.randn(100, 50) * 2.0,
        'uniform': torch.rand(100, 50) * 10.0 - 5.0,
        'sparse': torch.randn(100, 50) * 0.1,
        'outliers': torch.cat([torch.randn(90, 50) * 0.5, torch.randn(10, 50) * 5.0], dim=0)
    }
    
    quantization_schemes = [
        ('Symmetric Per-tensor', {'symmetric': True, 'per_channel': False}),
        ('Asymmetric Per-tensor', {'symmetric': False, 'per_channel': False}),
        ('Symmetric Per-channel', {'symmetric': True, 'per_channel': True, 'channel_axis': 1}),
        ('Asymmetric Per-channel', {'symmetric': False, 'per_channel': True, 'channel_axis': 1})
    ]
    
    for tensor_name, tensor in tensors.items():
        print(f"\n{'='*20} {tensor_name.upper()} TENSOR {'='*20}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        print(f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
        
        for scheme_name, scheme_params in quantization_schemes:
            print(f"\n--- {scheme_name} ---")
            
            # Create and configure quantizer
            quantizer = Int8Quantizer(**scheme_params)
            quantizer.calibrate(tensor)
            
            # Quantize and dequantize
            quantized = quantizer.quantize(tensor)
            dequantized = quantizer.dequantize(quantized)
            
            # Calculate error metrics
            mse = torch.mean((tensor - dequantized) ** 2).item()
            mae = torch.mean(torch.abs(tensor - dequantized)).item()
            max_error = torch.max(torch.abs(tensor - dequantized)).item()
            
            print(f"  MSE: {mse:.8f}")
            print(f"  MAE: {mae:.8f}")
            print(f"  Max Error: {max_error:.8f}")
            print(f"  Quantized range: [{quantized.min().item()}, {quantized.max().item()}]")


def demonstrate_calibration_effects():
    """Show how different calibration data affects quantization quality."""
    
    print("\n" + "=" * 60)
    print("CALIBRATION DATA EFFECTS DEMONSTRATION")
    print("=" * 60)
    
    # Create test tensor
    test_tensor = torch.randn(1000, 100) * 2.0
    
    # Different calibration datasets
    calibration_sets = {
        'full_data': test_tensor,
        'subset_10%': test_tensor[:100],
        'subset_1%': test_tensor[:10],
        'different_distribution': torch.randn(100, 100) * 1.0,
        'limited_range': torch.randn(100, 100) * 0.5
    }
    
    print(f"Test tensor stats:")
    print(f"  Shape: {test_tensor.shape}")
    print(f"  Range: [{test_tensor.min().item():.4f}, {test_tensor.max().item():.4f}]")
    print(f"  Mean: {test_tensor.mean().item():.4f}, Std: {test_tensor.std().item():.4f}")
    
    for cal_name, cal_data in calibration_sets.items():
        print(f"\n--- Calibration with {cal_name} ---")
        
        # Create fresh quantizer and calibrate
        quantizer = Int8Quantizer(symmetric=False, per_channel=False)
        quantizer.calibrate(cal_data)
        
        # Quantize test tensor
        quantized = quantizer.quantize(test_tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Calculate metrics
        mse = torch.mean((test_tensor - dequantized) ** 2).item()
        mae = torch.mean(torch.abs(test_tensor - dequantized)).item()
        
        params = quantizer.get_quantization_params()
        print(f"  Calibration data shape: {cal_data.shape}")
        print(f"  Scale: {params['scale']:.6f}")
        print(f"  Zero-point: {params['zero_point']}")
        print(f"  Test MSE: {mse:.8f}")
        print(f"  Test MAE: {mae:.8f}")


if __name__ == "__main__":
    print("Int8 Quantization Tool - Basic Examples")
    print("This script demonstrates basic tensor quantization capabilities.")
    
    # Run demonstrations
    demonstrate_tensor_quantization()
    demonstrate_calibration_effects()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Try running with different tensor sizes and distributions!")
