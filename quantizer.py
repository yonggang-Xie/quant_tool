import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import warnings


class Int8Quantizer:
    """
    Int8 Quantization implementation for PyTorch tensors and models.
    Supports both symmetric and asymmetric quantization schemes.
    """
    
    def __init__(self, 
                 symmetric: bool = True,
                 per_channel: bool = False,
                 channel_axis: int = 0):
        """
        Initialize the Int8 Quantizer.
        
        Args:
            symmetric: If True, use symmetric quantization [-128, 127]
                      If False, use asymmetric quantization with zero-point
            per_channel: If True, compute separate scales per channel
            channel_axis: Axis along which to compute per-channel quantization
        """
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.scale = None
        self.zero_point = None
        self.is_calibrated = False
        
    def calibrate(self, tensor: torch.Tensor) -> None:
        """
        Calibrate the quantizer using the provided tensor to compute
        optimal scale and zero-point parameters.
        
        Args:
            tensor: Input tensor for calibration
        """
        if self.per_channel:
            self._calibrate_per_channel(tensor)
        else:
            self._calibrate_per_tensor(tensor)
        
        self.is_calibrated = True
    
    def _calibrate_per_tensor(self, tensor: torch.Tensor) -> None:
        """Calibrate using per-tensor quantization."""
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        if self.symmetric:
            # Symmetric quantization: scale based on max absolute value
            abs_max = max(abs(tensor_min), abs(tensor_max))
            self.scale = abs_max / 127.0
            self.zero_point = 0
        else:
            # Asymmetric quantization: use full int8 range
            self.scale = (tensor_max - tensor_min) / 255.0
            self.zero_point = -128 - tensor_min / self.scale
            self.zero_point = int(round(self.zero_point))
            self.zero_point = max(-128, min(127, self.zero_point))
    
    def _calibrate_per_channel(self, tensor: torch.Tensor) -> None:
        """Calibrate using per-channel quantization."""
        # Move channel axis to the front for easier processing
        if self.channel_axis != 0:
            tensor = tensor.transpose(0, self.channel_axis)
        
        num_channels = tensor.shape[0]
        tensor_flat = tensor.view(num_channels, -1)
        
        tensor_min = tensor_flat.min(dim=1)[0]
        tensor_max = tensor_flat.max(dim=1)[0]
        
        if self.symmetric:
            abs_max = torch.max(torch.abs(tensor_min), torch.abs(tensor_max))
            self.scale = abs_max / 127.0
            self.zero_point = torch.zeros_like(self.scale, dtype=torch.int8)
        else:
            self.scale = (tensor_max - tensor_min) / 255.0
            zero_point = -128 - tensor_min / self.scale
            self.zero_point = torch.round(zero_point).clamp(-128, 127).to(torch.int8)
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a float32 tensor to int8.
        
        Args:
            tensor: Input float32 tensor
            
        Returns:
            Quantized int8 tensor
        """
        if not self.is_calibrated:
            warnings.warn("Quantizer not calibrated. Auto-calibrating with current tensor.")
            self.calibrate(tensor)
        
        if self.per_channel:
            return self._quantize_per_channel(tensor)
        else:
            return self._quantize_per_tensor(tensor)
    
    def _quantize_per_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize using per-tensor parameters."""
        if self.scale == 0:
            return torch.zeros_like(tensor, dtype=torch.int8)
        
        quantized = tensor / self.scale + self.zero_point
        quantized = torch.round(quantized)
        quantized = torch.clamp(quantized, -128, 127)
        return quantized.to(torch.int8)
    
    def _quantize_per_channel(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize using per-channel parameters."""
        original_shape = tensor.shape
        
        # Move channel axis to the front
        if self.channel_axis != 0:
            tensor = tensor.transpose(0, self.channel_axis)
        
        # Reshape for broadcasting
        scale_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
        scale = self.scale.view(scale_shape)
        zero_point = self.zero_point.view(scale_shape)
        
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        quantized = tensor / scale + zero_point
        quantized = torch.round(quantized)
        quantized = torch.clamp(quantized, -128, 127)
        
        # Move channel axis back to original position
        if self.channel_axis != 0:
            quantized = quantized.transpose(0, self.channel_axis)
        
        return quantized.to(torch.int8)
    
    def dequantize(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dequantize an int8 tensor back to float32.
        
        Args:
            quantized_tensor: Input int8 tensor
            
        Returns:
            Dequantized float32 tensor
        """
        if not self.is_calibrated:
            raise ValueError("Cannot dequantize without calibration parameters.")
        
        if self.per_channel:
            return self._dequantize_per_channel(quantized_tensor)
        else:
            return self._dequantize_per_tensor(quantized_tensor)
    
    def _dequantize_per_tensor(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize using per-tensor parameters."""
        dequantized = (quantized_tensor.float() - self.zero_point) * self.scale
        return dequantized
    
    def _dequantize_per_channel(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize using per-channel parameters."""
        original_shape = quantized_tensor.shape
        
        # Move channel axis to the front
        if self.channel_axis != 0:
            quantized_tensor = quantized_tensor.transpose(0, self.channel_axis)
        
        # Reshape for broadcasting
        scale_shape = [quantized_tensor.shape[0]] + [1] * (quantized_tensor.ndim - 1)
        scale = self.scale.view(scale_shape)
        zero_point = self.zero_point.view(scale_shape)
        
        dequantized = (quantized_tensor.float() - zero_point) * scale
        
        # Move channel axis back to original position
        if self.channel_axis != 0:
            dequantized = dequantized.transpose(0, self.channel_axis)
        
        return dequantized
    
    def get_quantization_params(self) -> dict:
        """Get the current quantization parameters."""
        return {
            'scale': self.scale,
            'zero_point': self.zero_point,
            'symmetric': self.symmetric,
            'per_channel': self.per_channel,
            'channel_axis': self.channel_axis,
            'is_calibrated': self.is_calibrated
        }
    
    def set_quantization_params(self, scale: Union[float, torch.Tensor], 
                               zero_point: Union[int, torch.Tensor]) -> None:
        """Set quantization parameters manually."""
        self.scale = scale
        self.zero_point = zero_point
        self.is_calibrated = True


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer that performs int8 matrix multiplication.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize float weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Quantizers for weights and activations
        self.weight_quantizer = Int8Quantizer(symmetric=True, per_channel=True, channel_axis=0)
        self.activation_quantizer = Int8Quantizer(symmetric=False, per_channel=False)
        
        # Quantized parameters (will be set during quantization)
        self.quantized_weight = None
        self.is_quantized = False
    
    def quantize_weights(self) -> None:
        """Quantize the layer weights."""
        self.weight_quantizer.calibrate(self.weight.data)
        self.quantized_weight = self.weight_quantizer.quantize(self.weight.data)
        self.is_quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_quantized:
            # Quantize input activations
            if not self.activation_quantizer.is_calibrated:
                self.activation_quantizer.calibrate(x)
            
            x_quantized = self.activation_quantizer.quantize(x)
            
            # Perform quantized matrix multiplication
            # Note: In practice, this would use specialized int8 GEMM kernels
            # For demonstration, we dequantize, compute, and requantize
            x_dequant = self.activation_quantizer.dequantize(x_quantized)
            w_dequant = self.weight_quantizer.dequantize(self.quantized_weight)
            
            output = torch.nn.functional.linear(x_dequant, w_dequant, self.bias)
            return output
        else:
            # Standard float32 computation
            return torch.nn.functional.linear(x, self.weight, self.bias)


class QuantizedConv2d(nn.Module):
    """
    Quantized 2D Convolution layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize float weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        # Quantizers
        self.weight_quantizer = Int8Quantizer(symmetric=True, per_channel=True, channel_axis=0)
        self.activation_quantizer = Int8Quantizer(symmetric=False, per_channel=False)
        
        self.quantized_weight = None
        self.is_quantized = False
    
    def quantize_weights(self) -> None:
        """Quantize the layer weights."""
        self.weight_quantizer.calibrate(self.weight.data)
        self.quantized_weight = self.weight_quantizer.quantize(self.weight.data)
        self.is_quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_quantized:
            # Quantize input activations
            if not self.activation_quantizer.is_calibrated:
                self.activation_quantizer.calibrate(x)
            
            x_quantized = self.activation_quantizer.quantize(x)
            
            # Perform quantized convolution
            x_dequant = self.activation_quantizer.dequantize(x_quantized)
            w_dequant = self.weight_quantizer.dequantize(self.quantized_weight)
            
            output = torch.nn.functional.conv2d(x_dequant, w_dequant, self.bias, 
                                              self.stride, self.padding)
            return output
        else:
            # Standard float32 computation
            return torch.nn.functional.conv2d(x, self.weight, self.bias, 
                                            self.stride, self.padding)


def quantize_model(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
    """
    Quantize a PyTorch model by replacing supported layers with quantized versions.
    
    Args:
        model: Input PyTorch model
        calibration_data: Data used for calibration
        
    Returns:
        Quantized model
    """
    # This is a simplified version - in practice, you'd need more sophisticated
    # model traversal and layer replacement logic
    
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with quantized version
                quantized_layer = QuantizedLinear(child.in_features, child.out_features, 
                                                child.bias is not None)
                quantized_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    quantized_layer.bias.data = child.bias.data.clone()
                quantized_layer.quantize_weights()
                setattr(module, name, quantized_layer)
            elif isinstance(child, nn.Conv2d):
                # Replace with quantized version
                quantized_layer = QuantizedConv2d(child.in_channels, child.out_channels,
                                                child.kernel_size[0], child.stride[0],
                                                child.padding[0], child.bias is not None)
                quantized_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    quantized_layer.bias.data = child.bias.data.clone()
                quantized_layer.quantize_weights()
                setattr(module, name, quantized_layer)
            else:
                replace_layers(child)
    
    # Create a copy of the model
    quantized_model = type(model)()
    quantized_model.load_state_dict(model.state_dict())
    
    # Replace layers with quantized versions
    replace_layers(quantized_model)
    
    # Calibrate activation quantizers with calibration data
    quantized_model.eval()
    with torch.no_grad():
        _ = quantized_model(calibration_data)
    
    return quantized_model
