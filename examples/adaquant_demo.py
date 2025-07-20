#!/usr/bin/env python3
"""
AdaQuant (Adaptive Quantization) demonstration example.
Shows how to use AdaQuant for optimized quantization parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import SimpleMLPClassifier, SimpleCNN, generate_sample_data
from advanced_quantizers import AdaQuantQuantizer, AdaQuantConfig, FastFineTuneFramework
from quantizer import Int8Quantizer
from utils import calculate_model_size, benchmark_inference, calculate_quantization_error


def demonstrate_adaquant_tensor():
    """Demonstrate AdaQuant on individual tensors."""
    print("=" * 60)
    print("ADAQUANT TENSOR-LEVEL DEMONSTRATION")
    print("=" * 60)
    
    # Create test tensor with challenging distribution
    torch.manual_seed(42)
    tensor = torch.randn(64, 128) * 3.0 + 1.0  # Wide range with offset
    
    print(f"Original tensor stats:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"  Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
    
    # Standard quantization
    print(f"\n--- Standard Quantization ---")
    standard_quantizer = Int8Quantizer(symmetric=False, per_channel=False)
    standard_quantizer.calibrate(tensor)
    standard_quantized = standard_quantizer.quantize(tensor)
    standard_dequantized = standard_quantizer.dequantize(standard_quantized)
    
    standard_error = calculate_quantization_error(tensor, standard_quantized, standard_dequantized)
    print(f"Standard Quantization Parameters:")
    print(f"  Scale: {standard_quantizer.scale:.6f}")
    print(f"  Zero-point: {standard_quantizer.zero_point}")
    print(f"Standard Quantization Error:")
    print(f"  MSE: {standard_error['mse']:.8f}")
    print(f"  MAE: {standard_error['mae']:.8f}")
    print(f"  Cosine Similarity: {standard_error['cosine_similarity']:.6f}")
    
    # AdaQuant quantization
    print(f"\n--- AdaQuant Quantization ---")
    config = AdaQuantConfig(
        num_iterations=500,  # Reduced for demo
        learning_rate=1e-2,
        patience=50
    )
    
    adaquant_quantizer = AdaQuantQuantizer(
        symmetric=False, 
        per_channel=False,
        config=config
    )
    
    # Initialize with standard calibration
    adaquant_quantizer.calibrate(tensor)
    
    print(f"Initial parameters:")
    print(f"  Scale: {adaquant_quantizer.scale:.6f}")
    print(f"  Zero-point: {adaquant_quantizer.zero_point}")
    
    # Optimize parameters
    print("Optimizing quantization parameters...")
    
    # Make parameters learnable
    scale_param = nn.Parameter(torch.tensor(adaquant_quantizer.scale))
    zp_param = nn.Parameter(torch.tensor(float(adaquant_quantizer.zero_point)))
    
    optimizer = torch.optim.Adam([scale_param, zp_param], 
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(config.num_iterations):
        optimizer.zero_grad()
        
        # Update quantization parameters
        current_scale = torch.clamp(scale_param, min=1e-8)
        current_zp = torch.round(zp_param).clamp(-128, 127)
        
        # Quantize with current parameters
        scaled_tensor = tensor / current_scale + current_zp
        quantized = torch.round(scaled_tensor).clamp(-128, 127)
        
        # Dequantize
        dequantized = (quantized - current_zp) * current_scale
        
        # Reconstruction loss
        loss = F.mse_loss(dequantized, tensor)
        
        loss.backward()
        
        # Constrain scale to be positive
        with torch.no_grad():
            scale_param.data = torch.clamp(scale_param.data, min=1e-8)
        
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss - config.early_stop_threshold:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            print(f"Early stopping at iteration {iteration}")
            break
            
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}, Loss: {loss.item():.6f}, "
                  f"Scale: {current_scale.item():.6f}, ZP: {current_zp.item():.0f}")
    
    # Update quantizer with optimized parameters
    adaquant_quantizer.scale = scale_param.item()
    adaquant_quantizer.zero_point = int(torch.round(zp_param).clamp(-128, 127).item())
    
    # Get final quantized result
    adaquant_quantized = adaquant_quantizer.quantize(tensor)
    adaquant_dequantized = adaquant_quantizer.dequantize(adaquant_quantized)
    
    adaquant_error = calculate_quantization_error(tensor, adaquant_quantized, adaquant_dequantized)
    print(f"\nOptimized Quantization Parameters:")
    print(f"  Scale: {adaquant_quantizer.scale:.6f}")
    print(f"  Zero-point: {adaquant_quantizer.zero_point}")
    print(f"AdaQuant Quantization Error:")
    print(f"  MSE: {adaquant_error['mse']:.8f}")
    print(f"  MAE: {adaquant_error['mae']:.8f}")
    print(f"  Cosine Similarity: {adaquant_error['cosine_similarity']:.6f}")
    
    # Comparison
    print(f"\n--- Improvement Analysis ---")
    mse_improvement = (standard_error['mse'] - adaquant_error['mse']) / standard_error['mse'] * 100
    mae_improvement = (standard_error['mae'] - adaquant_error['mae']) / standard_error['mae'] * 100
    
    print(f"MSE Improvement: {mse_improvement:.2f}%")
    print(f"MAE Improvement: {mae_improvement:.2f}%")
    
    # Parameter changes
    scale_change = (adaquant_quantizer.scale - standard_quantizer.scale) / standard_quantizer.scale * 100
    zp_change = adaquant_quantizer.zero_point - standard_quantizer.zero_point
    
    print(f"Scale change: {scale_change:+.2f}%")
    print(f"Zero-point change: {zp_change:+d}")


def demonstrate_adaquant_per_channel():
    """Demonstrate AdaQuant with per-channel quantization."""
    print("\n" + "=" * 60)
    print("ADAQUANT PER-CHANNEL DEMONSTRATION")
    print("=" * 60)
    
    # Create tensor with varying channel statistics
    torch.manual_seed(42)
    num_channels = 16
    tensor_data = []
    
    for i in range(num_channels):
        # Each channel has different scale and offset
        scale = 0.5 + i * 0.2
        offset = (i - num_channels//2) * 0.3
        channel = torch.randn(64) * scale + offset
        tensor_data.append(channel)
    
    tensor = torch.stack(tensor_data, dim=0)  # Shape: (16, 64)
    
    print(f"Created tensor with {num_channels} channels")
    print(f"Channel scales vary from 0.5 to {0.5 + (num_channels-1) * 0.2:.1f}")
    print(f"Overall range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    
    # Standard per-channel quantization
    print(f"\n--- Standard Per-channel Quantization ---")
    standard_quantizer = Int8Quantizer(symmetric=False, per_channel=True, channel_axis=0)
    standard_quantizer.calibrate(tensor)
    standard_quantized = standard_quantizer.quantize(tensor)
    standard_dequantized = standard_quantizer.dequantize(standard_quantized)
    
    standard_error = calculate_quantization_error(tensor, standard_quantized, standard_dequantized)
    print(f"Standard Per-channel Error:")
    print(f"  MSE: {standard_error['mse']:.8f}")
    print(f"  Scale range: [{standard_quantizer.scale.min().item():.6f}, {standard_quantizer.scale.max().item():.6f}]")
    
    # AdaQuant per-channel optimization
    print(f"\n--- AdaQuant Per-channel Optimization ---")
    config = AdaQuantConfig(
        num_iterations=300,
        learning_rate=1e-2,
        patience=30
    )
    
    adaquant_quantizer = AdaQuantQuantizer(
        symmetric=False, 
        per_channel=True,
        channel_axis=0,
        config=config
    )
    
    # Initialize
    adaquant_quantizer.calibrate(tensor)
    
    # Make parameters learnable
    scale_param = nn.Parameter(adaquant_quantizer.scale.clone())
    zp_param = nn.Parameter(adaquant_quantizer.zero_point.float().clone())
    
    optimizer = torch.optim.Adam([scale_param, zp_param], 
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    print("Optimizing per-channel parameters...")
    
    for iteration in range(config.num_iterations):
        optimizer.zero_grad()
        
        # Update parameters
        current_scale = torch.clamp(scale_param, min=1e-8)
        current_zp = torch.round(zp_param).clamp(-128, 127)
        
        # Per-channel quantization
        scale_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
        scale_broadcast = current_scale.view(scale_shape)
        zp_broadcast = current_zp.view(scale_shape)
        
        scaled_tensor = tensor / scale_broadcast + zp_broadcast
        quantized = torch.round(scaled_tensor).clamp(-128, 127)
        dequantized = (quantized - zp_broadcast) * scale_broadcast
        
        # Loss
        loss = F.mse_loss(dequantized, tensor)
        
        loss.backward()
        
        # Constrain scale
        with torch.no_grad():
            scale_param.data = torch.clamp(scale_param.data, min=1e-8)
        
        optimizer.step()
        
        if iteration % 50 == 0:
            print(f"  Iteration {iteration}, Loss: {loss.item():.6f}")
    
    # Update quantizer
    adaquant_quantizer.scale = scale_param.data.clone()
    adaquant_quantizer.zero_point = torch.round(zp_param.data).clamp(-128, 127).to(torch.int8)
    
    # Evaluate
    adaquant_quantized = adaquant_quantizer.quantize(tensor)
    adaquant_dequantized = adaquant_quantizer.dequantize(adaquant_quantized)
    
    adaquant_error = calculate_quantization_error(tensor, adaquant_quantized, adaquant_dequantized)
    print(f"\nAdaQuant Per-channel Error:")
    print(f"  MSE: {adaquant_error['mse']:.8f}")
    print(f"  Scale range: [{adaquant_quantizer.scale.min().item():.6f}, {adaquant_quantizer.scale.max().item():.6f}]")
    
    # Improvement
    improvement = (standard_error['mse'] - adaquant_error['mse']) / standard_error['mse'] * 100
    print(f"Per-channel MSE Improvement: {improvement:.2f}%")


def demonstrate_adaquant_model():
    """Demonstrate AdaQuant on a complete model."""
    print("\n" + "=" * 60)
    print("ADAQUANT MODEL-LEVEL DEMONSTRATION")
    print("=" * 60)
    
    # Create model
    torch.manual_seed(42)
    model = SimpleMLPClassifier(100, [64, 32], 10)
    
    # Generate data
    calibration_data = generate_sample_data('mlp', batch_size=50)
    test_data = generate_sample_data('mlp', batch_size=20)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Standard quantization baseline
    print(f"\n--- Standard Quantization Baseline ---")
    standard_errors = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizer = Int8Quantizer(symmetric=False, per_channel=True, channel_axis=0)
            quantizer.calibrate(module.weight.data)
            
            quantized_weight = quantizer.quantize(module.weight.data)
            dequantized_weight = quantizer.dequantize(quantized_weight)
            
            error = calculate_quantization_error(module.weight.data, quantized_weight, dequantized_weight)
            standard_errors.append(error['mse'])
            print(f"  Layer {name}: MSE = {error['mse']:.8f}")
    
    avg_standard_error = np.mean(standard_errors)
    print(f"Average Standard MSE: {avg_standard_error:.8f}")
    
    # AdaQuant optimization
    print(f"\n--- AdaQuant Model Optimization ---")
    
    config = AdaQuantConfig(
        num_iterations=200,  # Reduced for demo
        learning_rate=1e-2,
        patience=20
    )
    
    adaquant_errors = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nOptimizing layer: {name}")
            
            # Create AdaQuant quantizer
            quantizer = AdaQuantQuantizer(
                symmetric=False, 
                per_channel=True, 
                channel_axis=0,
                config=config
            )
            
            # Get layer input and output
            with torch.no_grad():
                if name == 'network.0':  # First layer
                    layer_input = calibration_data
                elif name == 'network.3':  # Second layer
                    layer_input = model.network[:3](calibration_data)
                elif name == 'network.6':  # Third layer
                    layer_input = model.network[:6](calibration_data)
                else:
                    continue
                
                original_output = module(layer_input)
            
            # Initialize quantizer
            quantizer.calibrate(module.weight.data)
            
            # Make parameters learnable
            scale_param = nn.Parameter(quantizer.scale.clone())
            zp_param = nn.Parameter(quantizer.zero_point.float().clone())
            
            optimizer = torch.optim.Adam([scale_param, zp_param], 
                                       lr=config.learning_rate,
                                       weight_decay=config.weight_decay)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Update parameters
                current_scale = torch.clamp(scale_param, min=1e-8)
                current_zp = torch.round(zp_param).clamp(-128, 127)
                
                # Quantize weights
                scale_shape = [module.weight.shape[0]] + [1] * (module.weight.ndim - 1)
                scale_broadcast = current_scale.view(scale_shape)
                zp_broadcast = current_zp.view(scale_shape)
                
                scaled_weights = module.weight.data / scale_broadcast + zp_broadcast
                quantized_weights = torch.round(scaled_weights).clamp(-128, 127)
                dequantized_weights = (quantized_weights - zp_broadcast) * scale_broadcast
                
                # Compute layer output with quantized weights
                quantized_output = F.linear(layer_input, dequantized_weights, module.bias)
                
                # Loss
                loss = F.mse_loss(quantized_output, original_output)
                
                loss.backward()
                
                # Constrain scale
                with torch.no_grad():
                    scale_param.data = torch.clamp(scale_param.data, min=1e-8)
                
                optimizer.step()
                
                # Early stopping
                if loss.item() < best_loss - config.early_stop_threshold:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= config.patience:
                    break
                    
                if iteration % 50 == 0:
                    print(f"    Iteration {iteration}, Loss: {loss.item():.6f}")
            
            # Update quantizer and evaluate
            quantizer.scale = scale_param.data.clone()
            quantizer.zero_point = torch.round(zp_param.data).clamp(-128, 127).to(torch.int8)
            
            final_quantized = quantizer.quantize(module.weight.data)
            final_dequantized = quantizer.dequantize(final_quantized)
            
            error = calculate_quantization_error(module.weight.data, final_quantized, final_dequantized)
            adaquant_errors.append(error['mse'])
            print(f"    Final MSE: {error['mse']:.8f}")
    
    avg_adaquant_error = np.mean(adaquant_errors)
    print(f"\nAverage AdaQuant MSE: {avg_adaquant_error:.8f}")
    
    # Overall improvement
    improvement = (avg_standard_error - avg_adaquant_error) / avg_standard_error * 100
    print(f"Overall MSE Improvement: {improvement:.2f}%")


def demonstrate_adaquant_vs_adaround():
    """Compare AdaQuant with AdaRound on the same tensor."""
    print("\n" + "=" * 60)
    print("ADAQUANT VS ADAROUND COMPARISON")
    print("=" * 60)
    
    # Create challenging tensor
    torch.manual_seed(42)
    tensor = torch.randn(32, 64) * 2.5 + 0.8
    
    print(f"Test tensor shape: {tensor.shape}")
    print(f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    
    methods = ['Standard', 'AdaQuant', 'AdaRound']
    results = {}
    
    for method in methods:
        print(f"\n--- {method} Quantization ---")
        
        if method == 'Standard':
            quantizer = Int8Quantizer(symmetric=False, per_channel=False)
            quantizer.calibrate(tensor)
            quantized = quantizer.quantize(tensor)
            dequantized = quantizer.dequantize(quantized)
            
        elif method == 'AdaQuant':
            from advanced_quantizers import AdaQuantQuantizer, AdaQuantConfig
            
            config = AdaQuantConfig(num_iterations=200, learning_rate=1e-2)
            quantizer = AdaQuantQuantizer(symmetric=False, per_channel=False, config=config)
            quantizer.calibrate(tensor)
            
            # Optimize parameters (simplified)
            scale_param = nn.Parameter(torch.tensor(quantizer.scale))
            zp_param = nn.Parameter(torch.tensor(float(quantizer.zero_point)))
            
            optimizer = torch.optim.Adam([scale_param, zp_param], lr=config.learning_rate)
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                current_scale = torch.clamp(scale_param, min=1e-8)
                current_zp = torch.round(zp_param).clamp(-128, 127)
                
                scaled_tensor = tensor / current_scale + current_zp
                quantized_temp = torch.round(scaled_tensor).clamp(-128, 127)
                dequantized_temp = (quantized_temp - current_zp) * current_scale
                
                loss = F.mse_loss(dequantized_temp, tensor)
                loss.backward()
                
                with torch.no_grad():
                    scale_param.data = torch.clamp(scale_param.data, min=1e-8)
                
                optimizer.step()
            
            quantizer.scale = scale_param.item()
            quantizer.zero_point = int(torch.round(zp_param).clamp(-128, 127).item())
            
            quantized = quantizer.quantize(tensor)
            dequantized = quantizer.dequantize(quantized)
            
        else:  # AdaRound
            from advanced_quantizers import AdaRoundQuantizer, AdaRoundConfig
            
            config = AdaRoundConfig(num_iterations=200, learning_rate=1e-3)
            quantizer = AdaRoundQuantizer(symmetric=False, per_channel=False, config=config)
            quantizer.calibrate(tensor)
            quantizer._initialize_rounding_params(tensor.shape)
            
            # Train rounding parameters (simplified)
            optimizer = torch.optim.Adam([quantizer.rounding_params], lr=config.learning_rate)
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                beta = 20.0
                soft_quantized = quantizer._soft_quantize(tensor, beta)
                dequantized_temp = (soft_quantized - quantizer.zero_point) * quantizer.scale
                
                loss = F.mse_loss(dequantized_temp, tensor)
                h = torch.sigmoid(beta * (quantizer.rounding_params - 0.5))
                reg = 0.01 * torch.sum(torch.min(h, 1 - h))
                
                total_loss = loss + reg
                total_loss.backward()
                optimizer.step()
            
            quantized = quantizer._hard_quantize(tensor)
            dequantized = quantizer.dequantize(quantized)
        
        # Calculate error
        error = calculate_quantization_error(tensor, quantized, dequantized)
        results[method] = error
        
        print(f"  MSE: {error['mse']:.8f}")
        print(f"  MAE: {error['mae']:.8f}")
        print(f"  Cosine Similarity: {error['cosine_similarity']:.6f}")
    
    # Comparison summary
    print(f"\n--- Method Comparison ---")
    baseline_mse = results['Standard']['mse']
    
    for method in methods:
        if method == 'Standard':
            print(f"{method:10}: MSE={results[method]['mse']:.8f} (baseline)")
        else:
            improvement = (baseline_mse - results[method]['mse']) / baseline_mse * 100
            print(f"{method:10}: MSE={results[method]['mse']:.8f} ({improvement:+.1f}%)")


if __name__ == "__main__":
    print("AdaQuant (Adaptive Quantization) Demonstration")
    print("This script shows how AdaQuant optimizes quantization parameters.")
    
    # Run demonstrations
    demonstrate_adaquant_tensor()
    demonstrate_adaquant_per_channel()
    demonstrate_adaquant_model()
    demonstrate_adaquant_vs_adaround()
    
    print("\n" + "=" * 60)
    print("ADAQUANT DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print("1. AdaQuant optimizes scale and zero-point parameters")
    print("2. Works well with both per-tensor and per-channel quantization")
    print("3. Gradient-based optimization for minimal reconstruction error")
    print("4. Complementary to AdaRound - can be used together")
