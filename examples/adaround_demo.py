#!/usr/bin/env python3
"""
AdaRound (Adaptive Rounding) demonstration example.
Shows how to use AdaRound for improved post-training quantization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from models import SimpleMLPClassifier, SimpleCNN, generate_sample_data
from advanced_quantizers import AdaRoundQuantizer, AdaRoundConfig, FastFineTuneFramework
from quantizer import Int8Quantizer
from utils import calculate_model_size, benchmark_inference, calculate_quantization_error


def demonstrate_adaround_tensor():
    """Demonstrate AdaRound on individual tensors."""
    print("=" * 60)
    print("ADAROUND TENSOR-LEVEL DEMONSTRATION")
    print("=" * 60)
    
    # Create test tensor with challenging distribution
    torch.manual_seed(42)
    tensor = torch.randn(64, 128) * 2.0 + 0.5  # Offset to make asymmetric
    
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
    print(f"Standard Quantization Error:")
    print(f"  MSE: {standard_error['mse']:.8f}")
    print(f"  MAE: {standard_error['mae']:.8f}")
    print(f"  Cosine Similarity: {standard_error['cosine_similarity']:.6f}")
    
    # AdaRound quantization
    print(f"\n--- AdaRound Quantization ---")
    config = AdaRoundConfig(
        num_iterations=2000,  # Reduced for demo
        learning_rate=1e-3,
        patience=200
    )
    
    adaround_quantizer = AdaRoundQuantizer(
        symmetric=False, 
        per_channel=False,
        config=config
    )
    
    # For tensor-level demo, we need to create a simple forward function
    def tensor_forward_fn(weights):
        # Simple identity function for demonstration
        return weights
    
    # Train AdaRound
    print("Training AdaRound parameters...")
    adaround_quantizer.calibrate(tensor)
    
    # Initialize rounding parameters
    adaround_quantizer._initialize_rounding_params(tensor.shape)
    
    # Train (simplified version)
    optimizer = torch.optim.Adam([adaround_quantizer.rounding_params], lr=config.learning_rate)
    
    for iteration in range(500):  # Reduced iterations for demo
        optimizer.zero_grad()
        
        beta = 20.0  # Fixed beta for simplicity
        soft_quantized = adaround_quantizer._soft_quantize(tensor, beta)
        
        # Dequantize
        scale, zero_point = adaround_quantizer.scale, adaround_quantizer.zero_point
        dequantized = (soft_quantized - zero_point) * scale
        
        # Loss (reconstruction error)
        loss = torch.nn.functional.mse_loss(dequantized, tensor)
        
        # Regularization
        h = torch.sigmoid(beta * (adaround_quantizer.rounding_params - 0.5))
        reg = 0.01 * torch.sum(torch.min(h, 1 - h))
        
        total_loss = loss + reg
        total_loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}, Loss: {total_loss.item():.6f}")
    
    # Get final quantized result
    adaround_quantized = adaround_quantizer._hard_quantize(tensor)
    adaround_dequantized = adaround_quantizer.dequantize(adaround_quantized)
    
    adaround_error = calculate_quantization_error(tensor, adaround_quantized, adaround_dequantized)
    print(f"\nAdaRound Quantization Error:")
    print(f"  MSE: {adaround_error['mse']:.8f}")
    print(f"  MAE: {adaround_error['mae']:.8f}")
    print(f"  Cosine Similarity: {adaround_error['cosine_similarity']:.6f}")
    
    # Comparison
    print(f"\n--- Improvement Analysis ---")
    mse_improvement = (standard_error['mse'] - adaround_error['mse']) / standard_error['mse'] * 100
    mae_improvement = (standard_error['mae'] - adaround_error['mae']) / standard_error['mae'] * 100
    
    print(f"MSE Improvement: {mse_improvement:.2f}%")
    print(f"MAE Improvement: {mae_improvement:.2f}%")
    
    # Analyze rounding decisions
    rounding_decisions = (adaround_quantizer.rounding_params > 0).float()
    print(f"Rounding decisions (up): {rounding_decisions.mean().item():.2%}")


def demonstrate_adaround_model():
    """Demonstrate AdaRound on a complete model."""
    print("\n" + "=" * 60)
    print("ADAROUND MODEL-LEVEL DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple model
    torch.manual_seed(42)
    model = SimpleMLPClassifier(784, [64, 32], 10)  # Fixed input size to match MNIST
    
    # Generate calibration data
    calibration_data = generate_sample_data('mlp', batch_size=50)
    test_data = generate_sample_data('mlp', batch_size=20)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get original model performance
    model.eval()
    with torch.no_grad():
        original_output = model(test_data)
    
    print(f"Original output range: [{original_output.min().item():.4f}, {original_output.max().item():.4f}]")
    
    # Standard quantization comparison
    print(f"\n--- Standard Quantization ---")
    standard_errors = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizer = Int8Quantizer(symmetric=True, per_channel=True, channel_axis=0)
            quantizer.calibrate(module.weight.data)
            
            quantized_weight = quantizer.quantize(module.weight.data)
            dequantized_weight = quantizer.dequantize(quantized_weight)
            
            error = calculate_quantization_error(module.weight.data, quantized_weight, dequantized_weight)
            standard_errors.append(error['mse'])
            print(f"  Layer {name}: MSE = {error['mse']:.8f}")
    
    avg_standard_error = np.mean(standard_errors)
    print(f"Average Standard MSE: {avg_standard_error:.8f}")
    
    # AdaRound with Fast Fine-tune Framework
    print(f"\n--- AdaRound Fine-tuning ---")
    
    config = AdaRoundConfig(
        num_iterations=1000,  # Reduced for demo
        learning_rate=1e-3,
        patience=100,
        early_stop_threshold=1e-6
    )
    
    framework = FastFineTuneFramework(method='adaround', config=config)
    
    # Fine-tune model (simplified version for demo)
    adaround_errors = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nProcessing layer: {name}")
            
            # Create AdaRound quantizer
            quantizer = AdaRoundQuantizer(
                symmetric=True, 
                per_channel=True, 
                channel_axis=0,
                config=config
            )
            
            # Get original layer output
            with torch.no_grad():
                layer_input = calibration_data
                if name == 'network.3':  # Second layer
                    layer_input = model.network[:3](calibration_data)
                elif name == 'network.6':  # Third layer  
                    layer_input = model.network[:6](calibration_data)
                
                original_layer_output = module(layer_input)
            
            # Define layer forward function
            def layer_forward_fn(weights):
                return torch.nn.functional.linear(layer_input, weights, module.bias)
            
            # Simplified AdaRound training
            quantizer.calibrate(module.weight.data)
            quantizer._initialize_rounding_params(module.weight.data.shape)
            
            optimizer = torch.optim.Adam([quantizer.rounding_params], lr=config.learning_rate)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(200):  # Reduced for demo
                optimizer.zero_grad()
                
                beta = 20.0  # Fixed beta
                soft_quantized = quantizer._soft_quantize(module.weight.data, beta)
                
                # Dequantize
                if quantizer.per_channel:
                    scale, zero_point = quantizer._get_per_channel_params(module.weight.data)
                else:
                    scale, zero_point = quantizer.scale, quantizer.zero_point
                
                dequantized_weights = (soft_quantized - zero_point) * scale
                
                # Compute layer output
                quantized_layer_output = layer_forward_fn(dequantized_weights)
                
                # Loss
                loss = torch.nn.functional.mse_loss(quantized_layer_output, original_layer_output)
                
                # Regularization
                h = torch.sigmoid(beta * (quantizer.rounding_params - 0.5))
                reg = 0.01 * torch.sum(torch.min(h, 1 - h))
                
                total_loss = loss + reg
                total_loss.backward()
                optimizer.step()
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 50:  # Early stopping
                    break
                    
                if iteration % 50 == 0:
                    print(f"    Iteration {iteration}, Loss: {total_loss.item():.6f}")
            
            # Evaluate final quantization
            final_quantized = quantizer._hard_quantize(module.weight.data)
            final_dequantized = quantizer.dequantize(final_quantized)
            
            error = calculate_quantization_error(module.weight.data, final_quantized, final_dequantized)
            adaround_errors.append(error['mse'])
            print(f"    Final MSE: {error['mse']:.8f}")
    
    avg_adaround_error = np.mean(adaround_errors)
    print(f"\nAverage AdaRound MSE: {avg_adaround_error:.8f}")
    
    # Overall improvement
    improvement = (avg_standard_error - avg_adaround_error) / avg_standard_error * 100
    print(f"Overall MSE Improvement: {improvement:.2f}%")


def demonstrate_adaround_comparison():
    """Compare AdaRound with different configurations."""
    print("\n" + "=" * 60)
    print("ADAROUND CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Create test tensor
    torch.manual_seed(42)
    tensor = torch.randn(32, 64) * 1.5
    
    configs = [
        ("Conservative", AdaRoundConfig(num_iterations=500, learning_rate=1e-4)),
        ("Standard", AdaRoundConfig(num_iterations=1000, learning_rate=1e-3)),
        ("Aggressive", AdaRoundConfig(num_iterations=2000, learning_rate=1e-2)),
    ]
    
    print(f"Testing tensor shape: {tensor.shape}")
    print(f"Original range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    
    results = []
    
    for config_name, config in configs:
        print(f"\n--- {config_name} Configuration ---")
        
        quantizer = AdaRoundQuantizer(symmetric=False, per_channel=False, config=config)
        quantizer.calibrate(tensor)
        quantizer._initialize_rounding_params(tensor.shape)
        
        # Simplified training
        optimizer = torch.optim.Adam([quantizer.rounding_params], lr=config.learning_rate)
        
        for iteration in range(min(config.num_iterations, 300)):  # Cap for demo
            optimizer.zero_grad()
            
            beta = 20.0
            soft_quantized = quantizer._soft_quantize(tensor, beta)
            dequantized = (soft_quantized - quantizer.zero_point) * quantizer.scale
            
            loss = torch.nn.functional.mse_loss(dequantized, tensor)
            h = torch.sigmoid(beta * (quantizer.rounding_params - 0.5))
            reg = 0.01 * torch.sum(torch.min(h, 1 - h))
            
            total_loss = loss + reg
            total_loss.backward()
            optimizer.step()
        
        # Evaluate
        final_quantized = quantizer._hard_quantize(tensor)
        final_dequantized = quantizer.dequantize(final_quantized)
        
        error = calculate_quantization_error(tensor, final_quantized, final_dequantized)
        
        results.append({
            'config': config_name,
            'mse': error['mse'],
            'mae': error['mae'],
            'cosine_sim': error['cosine_similarity']
        })
        
        print(f"  MSE: {error['mse']:.8f}")
        print(f"  MAE: {error['mae']:.8f}")
        print(f"  Cosine Similarity: {error['cosine_similarity']:.6f}")
    
    # Summary
    print(f"\n--- Configuration Summary ---")
    best_mse = min(r['mse'] for r in results)
    for result in results:
        improvement = (best_mse / result['mse'] - 1) * 100 if result['mse'] != best_mse else 0
        print(f"{result['config']:12}: MSE={result['mse']:.8f} "
              f"({'BEST' if result['mse'] == best_mse else f'{improvement:+.1f}%'})")


if __name__ == "__main__":
    print("AdaRound (Adaptive Rounding) Demonstration")
    print("This script shows how AdaRound improves quantization accuracy.")
    
    # Run demonstrations
    demonstrate_adaround_tensor()
    demonstrate_adaround_model()
    demonstrate_adaround_comparison()
    
    print("\n" + "=" * 60)
    print("ADAROUND DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print("1. AdaRound learns optimal rounding decisions per weight")
    print("2. Significant improvement over standard rounding")
    print("3. Post-training quantization without full retraining")
    print("4. Configurable for different accuracy/speed trade-offs")
