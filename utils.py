import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Any
try:
    from .quantizer import Int8Quantizer
except ImportError:
    from quantizer import Int8Quantizer


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the memory footprint of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'total_mb': total_size / (1024 ** 2)
    }


def benchmark_inference(model: nn.Module, input_tensor: torch.Tensor, 
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark inference time of a model.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for inference
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs (not counted in timing)
        
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times)
    }


def calculate_quantization_error(original_tensor: torch.Tensor, 
                               quantized_tensor: torch.Tensor,
                               dequantized_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various error metrics for quantization.
    
    Args:
        original_tensor: Original float32 tensor
        quantized_tensor: Quantized int8 tensor
        dequantized_tensor: Dequantized float32 tensor
        
    Returns:
        Dictionary with error metrics
    """
    # Mean Squared Error
    mse = torch.mean((original_tensor - dequantized_tensor) ** 2).item()
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(original_tensor - dequantized_tensor)).item()
    
    # Signal-to-Noise Ratio
    signal_power = torch.mean(original_tensor ** 2).item()
    noise_power = torch.mean((original_tensor - dequantized_tensor) ** 2).item()
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Peak Signal-to-Noise Ratio
    max_val = torch.max(torch.abs(original_tensor)).item()
    psnr_db = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-10))
    
    # Cosine Similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        original_tensor.flatten(), dequantized_tensor.flatten(), dim=0
    ).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'snr_db': snr_db,
        'psnr_db': psnr_db,
        'cosine_similarity': cos_sim,
        'relative_error': mae / (torch.mean(torch.abs(original_tensor)).item() + 1e-10)
    }


def analyze_tensor_distribution(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze the statistical distribution of a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with distribution statistics
    """
    tensor_flat = tensor.flatten()
    
    return {
        'mean': tensor_flat.mean().item(),
        'std': tensor_flat.std().item(),
        'min': tensor_flat.min().item(),
        'max': tensor_flat.max().item(),
        'median': tensor_flat.median().item(),
        'q25': tensor_flat.quantile(0.25).item(),
        'q75': tensor_flat.quantile(0.75).item(),
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'num_elements': tensor.numel()
    }


def compare_models(original_model: nn.Module, quantized_model: nn.Module,
                  test_input: torch.Tensor, num_classes: int = None) -> Dict[str, Any]:
    """
    Compare original and quantized models across multiple metrics.
    
    Args:
        original_model: Original float32 model
        quantized_model: Quantized int8 model
        test_input: Test input tensor
        num_classes: Number of classes for classification accuracy (optional)
        
    Returns:
        Comprehensive comparison results
    """
    results = {}
    
    # Model sizes
    results['original_size'] = calculate_model_size(original_model)
    results['quantized_size'] = calculate_model_size(quantized_model)
    results['size_reduction_ratio'] = (
        results['original_size']['total_mb'] / results['quantized_size']['total_mb']
    )
    
    # Inference timing
    results['original_timing'] = benchmark_inference(original_model, test_input)
    results['quantized_timing'] = benchmark_inference(quantized_model, test_input)
    results['speedup_ratio'] = (
        results['original_timing']['mean_ms'] / results['quantized_timing']['mean_ms']
    )
    
    # Output comparison
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        quantized_output = quantized_model(test_input)
    
    results['output_error'] = calculate_quantization_error(
        original_output, None, quantized_output
    )
    
    # Classification accuracy comparison (if applicable)
    if num_classes is not None and len(original_output.shape) == 2:
        original_pred = torch.argmax(original_output, dim=1)
        quantized_pred = torch.argmax(quantized_output, dim=1)
        
        # Agreement between predictions
        agreement = (original_pred == quantized_pred).float().mean().item()
        results['prediction_agreement'] = agreement
        
        # Top-k accuracy comparison
        for k in [1, 3, 5]:
            if k <= num_classes:
                orig_topk = torch.topk(original_output, k, dim=1)[1]
                quant_topk = torch.topk(quantized_output, k, dim=1)[1]
                
                # Check if top-k sets overlap
                overlap = 0
                for i in range(test_input.shape[0]):
                    orig_set = set(orig_topk[i].tolist())
                    quant_set = set(quant_topk[i].tolist())
                    overlap += len(orig_set.intersection(quant_set)) / k
                
                results[f'top{k}_overlap'] = overlap / test_input.shape[0]
    
    return results


def save_quantization_config(quantizer: Int8Quantizer, filepath: str) -> None:
    """
    Save quantization configuration to a file.
    
    Args:
        quantizer: Configured Int8Quantizer instance
        filepath: Path to save the configuration
    """
    config = quantizer.get_quantization_params()
    
    # Convert tensors to lists for JSON serialization
    if isinstance(config['scale'], torch.Tensor):
        config['scale'] = config['scale'].tolist()
    if isinstance(config['zero_point'], torch.Tensor):
        config['zero_point'] = config['zero_point'].tolist()
    
    import json
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_quantization_config(filepath: str) -> Dict[str, Any]:
    """
    Load quantization configuration from a file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    # Convert lists back to tensors if needed
    if isinstance(config['scale'], list):
        config['scale'] = torch.tensor(config['scale'])
    if isinstance(config['zero_point'], list):
        config['zero_point'] = torch.tensor(config['zero_point'])
    
    return config


def visualize_quantization_effects(original_tensor: torch.Tensor,
                                 quantized_tensor: torch.Tensor,
                                 dequantized_tensor: torch.Tensor,
                                 save_path: str = None) -> None:
    """
    Create visualizations showing the effects of quantization.
    
    Args:
        original_tensor: Original float32 tensor
        quantized_tensor: Quantized int8 tensor  
        dequantized_tensor: Dequantized float32 tensor
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Flatten tensors for plotting
        orig_flat = original_tensor.flatten().numpy()
        quant_flat = quantized_tensor.flatten().numpy()
        dequant_flat = dequantized_tensor.flatten().numpy()
        
        # Original vs Dequantized scatter plot
        axes[0, 0].scatter(orig_flat, dequant_flat, alpha=0.5, s=1)
        axes[0, 0].plot([orig_flat.min(), orig_flat.max()], 
                       [orig_flat.min(), orig_flat.max()], 'r--')
        axes[0, 0].set_xlabel('Original Values')
        axes[0, 0].set_ylabel('Dequantized Values')
        axes[0, 0].set_title('Original vs Dequantized')
        
        # Quantization error histogram
        error = orig_flat - dequant_flat
        axes[0, 1].hist(error, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Quantization Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Quantization Error Distribution')
        
        # Original distribution
        axes[1, 0].hist(orig_flat, bins=50, alpha=0.7, label='Original')
        axes[1, 0].hist(dequant_flat, bins=50, alpha=0.7, label='Dequantized')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Value Distributions')
        axes[1, 0].legend()
        
        # Quantized values distribution
        axes[1, 1].hist(quant_flat, bins=range(-128, 129), alpha=0.7)
        axes[1, 1].set_xlabel('Quantized Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Quantized Values Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


def print_quantization_summary(quantizer: Int8Quantizer, tensor: torch.Tensor) -> None:
    """
    Print a summary of quantization parameters and effects.
    
    Args:
        quantizer: Configured Int8Quantizer instance
        tensor: Original tensor that was quantized
    """
    print("=" * 50)
    print("QUANTIZATION SUMMARY")
    print("=" * 50)
    
    params = quantizer.get_quantization_params()
    
    print(f"Quantization Type: {'Symmetric' if params['symmetric'] else 'Asymmetric'}")
    print(f"Per-channel: {params['per_channel']}")
    
    if params['per_channel']:
        print(f"Channel axis: {params['channel_axis']}")
        print(f"Number of channels: {len(params['scale'])}")
        print(f"Scale range: [{params['scale'].min().item():.6f}, {params['scale'].max().item():.6f}]")
        if not params['symmetric']:
            print(f"Zero-point range: [{params['zero_point'].min().item()}, {params['zero_point'].max().item()}]")
    else:
        print(f"Scale: {params['scale']:.6f}")
        print(f"Zero-point: {params['zero_point']}")
    
    # Quantize and analyze
    quantized = quantizer.quantize(tensor)
    dequantized = quantizer.dequantize(quantized)
    
    error_metrics = calculate_quantization_error(tensor, quantized, dequantized)
    
    print(f"\nTensor shape: {list(tensor.shape)}")
    print(f"Original range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
    print(f"Quantized range: [{quantized.min().item()}, {quantized.max().item()}]")
    print(f"Dequantized range: [{dequantized.min().item():.6f}, {dequantized.max().item():.6f}]")
    
    print(f"\nError Metrics:")
    print(f"  MSE: {error_metrics['mse']:.8f}")
    print(f"  MAE: {error_metrics['mae']:.8f}")
    print(f"  SNR: {error_metrics['snr_db']:.2f} dB")
    print(f"  PSNR: {error_metrics['psnr_db']:.2f} dB")
    print(f"  Cosine Similarity: {error_metrics['cosine_similarity']:.6f}")
    print(f"  Relative Error: {error_metrics['relative_error']:.6f}")
    
    print("=" * 50)
