#!/usr/bin/env python3
"""
Model quantization example comparing original vs quantized models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models import SimpleMLPClassifier, SimpleCNN, ModelFactory, generate_sample_data
from quantizer import Int8Quantizer, QuantizedLinear, QuantizedConv2d
from utils import compare_models, calculate_model_size, benchmark_inference


def quantize_mlp_model(model, calibration_data):
    """Manually quantize an MLP model by replacing layers."""
    # Extract architecture from original model
    hidden_sizes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'network' in name:
            # Skip the last layer (output layer)
            if int(name.split('.')[1]) < len(model.network) - 1:
                hidden_sizes.append(module.out_features)
    
    quantized_model = SimpleMLPClassifier(
        model.input_size, 
        hidden_sizes,
        model.num_classes
    )
    
    # Copy original weights
    quantized_model.load_state_dict(model.state_dict())
    
    # Replace linear layers with quantized versions
    layers = []
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            quant_layer = QuantizedLinear(layer.in_features, layer.out_features, 
                                        layer.bias is not None)
            quant_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                quant_layer.bias.data = layer.bias.data.clone()
            quant_layer.quantize_weights()
            layers.append(quant_layer)
        else:
            layers.append(layer)
    
    quantized_model.network = nn.Sequential(*layers)
    
    # Calibrate with sample data
    quantized_model.eval()
    with torch.no_grad():
        _ = quantized_model(calibration_data)
    
    return quantized_model


def demonstrate_mlp_quantization():
    """Demonstrate MLP quantization with performance comparison."""
    
    print("=" * 60)
    print("MLP MODEL QUANTIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create original model
    original_model = SimpleMLPClassifier(784, [256, 128, 64], 10)
    
    # Generate sample data
    calibration_data = generate_sample_data('mlp', batch_size=100)
    test_data = generate_sample_data('mlp', batch_size=32)
    
    print("Original Model:")
    original_size = calculate_model_size(original_model)
    print(f"  Parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"  Size: {original_size['total_mb']:.2f} MB")
    
    # Quantize model
    print("\nQuantizing model...")
    quantized_model = quantize_mlp_model(original_model, calibration_data)
    
    print("Quantized Model:")
    quantized_size = calculate_model_size(quantized_model)
    print(f"  Parameters: {sum(p.numel() for p in quantized_model.parameters()):,}")
    print(f"  Size: {quantized_size['total_mb']:.2f} MB")
    print(f"  Size reduction: {original_size['total_mb'] / quantized_size['total_mb']:.2f}x")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    original_timing = benchmark_inference(original_model, test_data, num_runs=50)
    quantized_timing = benchmark_inference(quantized_model, test_data, num_runs=50)
    
    print(f"  Original inference time: {original_timing['mean_ms']:.2f} ± {original_timing['std_ms']:.2f} ms")
    print(f"  Quantized inference time: {quantized_timing['mean_ms']:.2f} ± {quantized_timing['std_ms']:.2f} ms")
    print(f"  Speedup: {original_timing['mean_ms'] / quantized_timing['mean_ms']:.2f}x")
    
    # Output comparison
    print("\nOutput Comparison:")
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_data)
        quantized_output = quantized_model(test_data)
    
    mse = torch.mean((original_output - quantized_output) ** 2).item()
    mae = torch.mean(torch.abs(original_output - quantized_output)).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        original_output.flatten(), quantized_output.flatten(), dim=0
    ).item()
    
    print(f"  MSE: {mse:.8f}")
    print(f"  MAE: {mae:.8f}")
    print(f"  Cosine Similarity: {cos_sim:.6f}")


def demonstrate_different_quantization_schemes():
    """Compare different quantization schemes on the same model."""
    
    print("\n" + "=" * 60)
    print("QUANTIZATION SCHEMES COMPARISON")
    print("=" * 60)
    
    # Create a simple model for testing
    model = SimpleMLPClassifier(100, [64, 32], 10)
    test_data = torch.randn(32, 100)
    
    # Get original output
    model.eval()
    with torch.no_grad():
        original_output = model(test_data)
    
    # Test different quantization schemes
    schemes = [
        ('Symmetric Per-tensor', {'symmetric': True, 'per_channel': False}),
        ('Asymmetric Per-tensor', {'symmetric': False, 'per_channel': False}),
        ('Symmetric Per-channel', {'symmetric': True, 'per_channel': True, 'channel_axis': 0}),
        ('Asymmetric Per-channel', {'symmetric': False, 'per_channel': True, 'channel_axis': 0})
    ]
    
    print(f"Testing {len(schemes)} quantization schemes on MLP model")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for scheme_name, scheme_params in schemes:
        print(f"\n--- {scheme_name} ---")
        
        # Quantize each layer's weights
        total_error = 0
        num_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantizer = Int8Quantizer(**scheme_params)
                
                # Quantize weights
                original_weight = module.weight.data.clone()
                quantizer.calibrate(original_weight)
                quantized_weight = quantizer.quantize(original_weight)
                dequantized_weight = quantizer.dequantize(quantized_weight)
                
                # Calculate error for this layer
                layer_mse = torch.mean((original_weight - dequantized_weight) ** 2).item()
                total_error += layer_mse
                num_layers += 1
                
                print(f"  Layer {name}: MSE = {layer_mse:.8f}")
        
        avg_error = total_error / num_layers if num_layers > 0 else 0
        print(f"  Average layer MSE: {avg_error:.8f}")


def demonstrate_quantization_aware_training():
    """Demonstrate quantization-aware training concepts."""
    
    print("\n" + "=" * 60)
    print("QUANTIZATION-AWARE TRAINING SIMULATION")
    print("=" * 60)
    
    # Create model and data
    model = SimpleMLPClassifier(50, [32, 16], 5)
    train_data = torch.randn(100, 50)
    train_labels = torch.randint(0, 5, (100,))
    
    print("Simulating quantization-aware training...")
    print("(In practice, this would involve training with quantization in the loop)")
    
    # Simulate training with quantization noise
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining with quantization simulation:")
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_data)
        loss = criterion(output, train_labels)
        
        # Add quantization noise to gradients (simplified simulation)
        for param in model.parameters():
            if param.grad is not None:
                # Simulate quantization noise
                noise = torch.randn_like(param.grad) * 0.01
                param.grad += noise
        
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("\nQuantization-aware training helps models adapt to quantization errors")
    print("during training, leading to better post-quantization performance.")


def analyze_layer_sensitivity():
    """Analyze which layers are most sensitive to quantization."""
    
    print("\n" + "=" * 60)
    print("LAYER QUANTIZATION SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Create a deeper model for analysis
    model = SimpleMLPClassifier(100, [128, 64, 32, 16], 10)
    test_data = torch.randn(50, 100)
    
    # Get baseline output
    model.eval()
    with torch.no_grad():
        baseline_output = model(test_data)
    
    print("Analyzing sensitivity of each layer to quantization...")
    
    layer_sensitivities = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nTesting layer: {name}")
            
            # Store original weights
            original_weights = module.weight.data.clone()
            original_bias = module.bias.data.clone() if module.bias is not None else None
            
            # Quantize this layer only
            quantizer = Int8Quantizer(symmetric=False, per_channel=False)
            quantizer.calibrate(original_weights)
            quantized_weights = quantizer.quantize(original_weights)
            dequantized_weights = quantizer.dequantize(quantized_weights)
            
            # Replace weights temporarily
            module.weight.data = dequantized_weights
            
            # Test output change
            with torch.no_grad():
                modified_output = model(test_data)
            
            # Calculate sensitivity metrics
            output_mse = torch.mean((baseline_output - modified_output) ** 2).item()
            weight_mse = torch.mean((original_weights - dequantized_weights) ** 2).item()
            
            sensitivity = output_mse / (weight_mse + 1e-10)  # Avoid division by zero
            layer_sensitivities[name] = {
                'output_mse': output_mse,
                'weight_mse': weight_mse,
                'sensitivity': sensitivity
            }
            
            print(f"  Weight MSE: {weight_mse:.8f}")
            print(f"  Output MSE: {output_mse:.8f}")
            print(f"  Sensitivity: {sensitivity:.2f}")
            
            # Restore original weights
            module.weight.data = original_weights
            if original_bias is not None:
                module.bias.data = original_bias
    
    # Summary
    print(f"\n--- Sensitivity Summary ---")
    sorted_layers = sorted(layer_sensitivities.items(), 
                          key=lambda x: x[1]['sensitivity'], reverse=True)
    
    print("Most sensitive layers (highest impact on output):")
    for name, metrics in sorted_layers[:3]:
        print(f"  {name}: Sensitivity = {metrics['sensitivity']:.2f}")
    
    print("\nLeast sensitive layers (lowest impact on output):")
    for name, metrics in sorted_layers[-3:]:
        print(f"  {name}: Sensitivity = {metrics['sensitivity']:.2f}")


if __name__ == "__main__":
    print("Int8 Quantization Tool - Model Comparison Examples")
    print("This script demonstrates model-level quantization and analysis.")
    
    # Run demonstrations
    demonstrate_mlp_quantization()
    demonstrate_different_quantization_schemes()
    demonstrate_quantization_aware_training()
    analyze_layer_sensitivity()
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)
    print("These examples show various aspects of neural network quantization.")
    print("Try modifying the models and parameters to see different results!")
