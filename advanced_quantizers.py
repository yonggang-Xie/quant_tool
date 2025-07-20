import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
import warnings
from dataclasses import dataclass
from quantizer import Int8Quantizer


@dataclass
class AdaRoundConfig:
    """Configuration for AdaRound quantization."""
    num_iterations: int = 10000
    learning_rate: float = 1e-3
    beta_range: Tuple[float, float] = (20, 2)  # (start_beta, end_beta)
    warm_up: float = 0.2  # Fraction of iterations for warm-up
    weight_decay: float = 1e-5
    early_stop_threshold: float = 1e-6
    patience: int = 500
    

@dataclass
class AdaQuantConfig:
    """Configuration for AdaQuant quantization."""
    num_iterations: int = 1000
    learning_rate: float = 1e-2
    early_stop_threshold: float = 1e-5
    patience: int = 100
    weight_decay: float = 1e-4
    layer_wise_lr_factor: float = 1.0
    selective_update: bool = True


class AdaRoundQuantizer(Int8Quantizer):
    """
    AdaRound (Adaptive Rounding) Quantizer.
    
    Implements the adaptive rounding technique from:
    "Up or Down? Adaptive Rounding for Post-Training Quantization"
    https://arxiv.org/abs/2004.10568
    """
    
    def __init__(self, 
                 symmetric: bool = True,
                 per_channel: bool = False,
                 channel_axis: int = 0,
                 config: Optional[AdaRoundConfig] = None):
        super().__init__(symmetric, per_channel, channel_axis)
        self.config = config or AdaRoundConfig()
        self.rounding_params = None
        self.is_trained = False
        
    def _initialize_rounding_params(self, tensor_shape: torch.Size) -> None:
        """Initialize learnable rounding parameters."""
        if self.per_channel:
            if self.channel_axis == 0:
                param_shape = (tensor_shape[0],) + (1,) * (len(tensor_shape) - 1)
            else:
                param_shape = tensor_shape
        else:
            param_shape = tensor_shape
            
        # Initialize rounding parameters close to 0.5 (neutral rounding)
        self.rounding_params = nn.Parameter(
            torch.zeros(param_shape) + 0.01 * torch.randn(param_shape)
        )
    
    def _soft_quantize(self, tensor: torch.Tensor, beta: float) -> torch.Tensor:
        """Soft quantization using sigmoid for differentiable rounding."""
        if not self.is_calibrated:
            self.calibrate(tensor)
            
        if self.rounding_params is None:
            self._initialize_rounding_params(tensor.shape)
            
        # Get quantization parameters
        if self.per_channel:
            scale, zero_point = self._get_per_channel_params(tensor)
        else:
            scale, zero_point = self.scale, self.zero_point
            
        # Compute floor values
        scaled_tensor = tensor / scale + zero_point
        floor_vals = torch.floor(scaled_tensor)
        
        # Soft rounding using sigmoid
        h = torch.sigmoid(beta * (self.rounding_params - 0.5))
        soft_quantized = floor_vals + h
        
        # Clamp to valid range
        soft_quantized = torch.clamp(soft_quantized, -128, 127)
        
        return soft_quantized
    
    def _hard_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Hard quantization for inference."""
        if not self.is_calibrated:
            self.calibrate(tensor)
            
        if self.rounding_params is None:
            return super().quantize(tensor)
            
        # Get quantization parameters
        if self.per_channel:
            scale, zero_point = self._get_per_channel_params(tensor)
        else:
            scale, zero_point = self.scale, self.zero_point
            
        # Compute floor values
        scaled_tensor = tensor / scale + zero_point
        floor_vals = torch.floor(scaled_tensor)
        
        # Hard rounding based on learned parameters
        h = (self.rounding_params > 0).float()
        hard_quantized = floor_vals + h
        
        # Clamp to valid range
        hard_quantized = torch.clamp(hard_quantized, -128, 127)
        
        return hard_quantized.to(torch.int8)
    
    def _get_per_channel_params(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-channel scale and zero-point parameters."""
        if self.channel_axis != 0:
            tensor = tensor.transpose(0, self.channel_axis)
        
        scale_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
        scale = self.scale.view(scale_shape)
        zero_point = self.zero_point.view(scale_shape)
        
        return scale, zero_point
    
    def quantize(self, tensor: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Quantize tensor with adaptive rounding."""
        if training and self.rounding_params is not None:
            # Use soft quantization during training
            beta = self._get_current_beta(0)  # Default beta for inference
            return self._soft_quantize(tensor, beta)
        else:
            # Use hard quantization for inference
            return self._hard_quantize(tensor)
    
    def _get_current_beta(self, iteration: int) -> float:
        """Get current beta value for annealing."""
        progress = min(iteration / self.config.num_iterations, 1.0)
        beta_start, beta_end = self.config.beta_range
        return beta_start + (beta_end - beta_start) * progress
    
    def train_rounding(self, 
                      original_tensor: torch.Tensor,
                      target_output: torch.Tensor,
                      layer_forward_fn: callable) -> Dict[str, float]:
        """
        Train adaptive rounding parameters.
        
        Args:
            original_tensor: Original float weights
            target_output: Target output from original layer
            layer_forward_fn: Function to compute layer output
            
        Returns:
            Training statistics
        """
        if not self.is_calibrated:
            self.calibrate(original_tensor)
            
        if self.rounding_params is None:
            self._initialize_rounding_params(original_tensor.shape)
            
        # Setup optimizer
        optimizer = torch.optim.Adam([self.rounding_params], 
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        
        best_loss = float('inf')
        patience_counter = 0
        losses = []
        
        for iteration in range(self.config.num_iterations):
            optimizer.zero_grad()
            
            # Get current beta for annealing
            beta = self._get_current_beta(iteration)
            
            # Soft quantize weights
            soft_quantized = self._soft_quantize(original_tensor, beta)
            
            # Dequantize for layer computation
            if self.per_channel:
                scale, zero_point = self._get_per_channel_params(original_tensor)
            else:
                scale, zero_point = self.scale, self.zero_point
                
            dequantized_weights = (soft_quantized - zero_point) * scale
            
            # Compute layer output with quantized weights
            quantized_output = layer_forward_fn(dequantized_weights)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(quantized_output, target_output)
            
            # Regularization term (encourage binary decisions)
            reg_term = self._compute_regularization(beta)
            
            total_loss = recon_loss + reg_term
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            # Early stopping
            if total_loss.item() < best_loss - self.config.early_stop_threshold:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping at iteration {iteration}")
                break
                
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.6f}, "
                      f"Recon: {recon_loss.item():.6f}, Reg: {reg_term.item():.6f}")
        
        self.is_trained = True
        
        return {
            'final_loss': losses[-1] if losses else float('inf'),
            'best_loss': best_loss,
            'num_iterations': len(losses),
            'converged': patience_counter >= self.config.patience
        }
    
    def _compute_regularization(self, beta: float) -> torch.Tensor:
        """Compute regularization term to encourage binary decisions."""
        # Encourage rounding parameters to be close to 0 or 1
        h = torch.sigmoid(beta * (self.rounding_params - 0.5))
        reg = torch.sum(torch.min(h, 1 - h))
        return 0.01 * reg  # Small weight for regularization


class AdaQuantQuantizer(Int8Quantizer):
    """
    AdaQuant (Adaptive Quantization) Quantizer.
    
    Implements adaptive quantization with layer-wise parameter optimization.
    Based on principles from quantization literature for optimal parameter search.
    """
    
    def __init__(self,
                 symmetric: bool = True,
                 per_channel: bool = False,
                 channel_axis: int = 0,
                 config: Optional[AdaQuantConfig] = None):
        super().__init__(symmetric, per_channel, channel_axis)
        self.config = config or AdaQuantConfig()
        self.optimized_params = None
        self.is_optimized = False
        
    def optimize_parameters(self,
                          weights: torch.Tensor,
                          activations: torch.Tensor,
                          target_output: torch.Tensor,
                          layer_forward_fn: callable) -> Dict[str, float]:
        """
        Optimize quantization parameters using gradient-based search.
        
        Args:
            weights: Layer weights
            activations: Layer input activations
            target_output: Target output from original layer
            layer_forward_fn: Function to compute layer output
            
        Returns:
            Optimization statistics
        """
        # Initialize parameters
        if self.per_channel:
            self._initialize_per_channel_params(weights)
        else:
            self._initialize_per_tensor_params(weights)
            
        # Make parameters learnable
        if self.per_channel:
            scale_param = nn.Parameter(self.scale.clone())
            if not self.symmetric:
                zp_param = nn.Parameter(self.zero_point.float().clone())
        else:
            scale_param = nn.Parameter(torch.tensor(self.scale))
            if not self.symmetric:
                zp_param = nn.Parameter(torch.tensor(float(self.zero_point)))
                
        # Setup optimizer
        params = [scale_param]
        if not self.symmetric:
            params.append(zp_param)
            
        optimizer = torch.optim.Adam(params, 
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        
        best_loss = float('inf')
        patience_counter = 0
        losses = []
        
        for iteration in range(self.config.num_iterations):
            optimizer.zero_grad()
            
            # Update quantization parameters
            self.scale = scale_param.data.clone()
            if not self.symmetric:
                if self.per_channel:
                    self.zero_point = torch.round(zp_param.data).clamp(-128, 127).to(torch.int8)
                else:
                    self.zero_point = int(torch.round(zp_param.data).clamp(-128, 127).item())
            
            # Quantize weights
            quantized_weights = self.quantize(weights)
            dequantized_weights = self.dequantize(quantized_weights)
            
            # Compute layer output
            quantized_output = layer_forward_fn(dequantized_weights, activations)
            
            # Reconstruction loss
            loss = F.mse_loss(quantized_output, target_output)
            
            loss.backward()
            
            # Constrain scale to be positive
            with torch.no_grad():
                scale_param.data = torch.clamp(scale_param.data, min=1e-8)
                
            optimizer.step()
            
            losses.append(loss.item())
            
            # Early stopping
            if loss.item() < best_loss - self.config.early_stop_threshold:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping at iteration {iteration}")
                break
                
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        # Store optimized parameters
        self.optimized_params = {
            'scale': self.scale,
            'zero_point': self.zero_point,
            'final_loss': losses[-1] if losses else float('inf')
        }
        
        self.is_optimized = True
        
        return {
            'final_loss': losses[-1] if losses else float('inf'),
            'best_loss': best_loss,
            'num_iterations': len(losses),
            'converged': patience_counter >= self.config.patience
        }
    
    def _initialize_per_channel_params(self, tensor: torch.Tensor) -> None:
        """Initialize per-channel parameters for optimization."""
        self._calibrate_per_channel(tensor)
        
    def _initialize_per_tensor_params(self, tensor: torch.Tensor) -> None:
        """Initialize per-tensor parameters for optimization."""
        self._calibrate_per_tensor(tensor)


class FastFineTuneFramework:
    """
    Unified framework for AdaRound and AdaQuant fine-tuning.
    
    Implements the unified approach described in Quark documentation
    with advanced features like early stopping and selective updates.
    """
    
    def __init__(self, 
                 method: str = 'adaround',
                 config: Optional[Union[AdaRoundConfig, AdaQuantConfig]] = None):
        """
        Initialize the fast fine-tune framework.
        
        Args:
            method: 'adaround' or 'adaquant'
            config: Configuration for the chosen method
        """
        self.method = method.lower()
        if self.method == 'adaround':
            self.config = config or AdaRoundConfig()
        elif self.method == 'adaquant':
            self.config = config or AdaQuantConfig()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.layer_stats = {}
        
    def finetune_model(self, 
                      model: nn.Module,
                      calibration_data: torch.Tensor,
                      validation_data: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Fine-tune entire model using the selected method.
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Data for calibration and training
            validation_data: Optional validation data for selective updates
            
        Returns:
            Fine-tuning statistics
        """
        print(f"Starting {self.method.upper()} fine-tuning...")
        
        total_stats = {
            'method': self.method,
            'layers_processed': 0,
            'layers_improved': 0,
            'total_time': 0,
            'layer_stats': {}
        }
        
        # Process each quantizable layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                print(f"\nProcessing layer: {name}")
                
                # Create appropriate quantizer
                if self.method == 'adaround':
                    quantizer = AdaRoundQuantizer(config=self.config)
                else:
                    quantizer = AdaQuantQuantizer(config=self.config)
                
                # Fine-tune this layer
                layer_stats = self._finetune_layer(
                    module, quantizer, calibration_data, validation_data
                )
                
                total_stats['layer_stats'][name] = layer_stats
                total_stats['layers_processed'] += 1
                
                if layer_stats.get('improved', False):
                    total_stats['layers_improved'] += 1
                    
        print(f"\nFine-tuning complete!")
        print(f"Processed {total_stats['layers_processed']} layers")
        print(f"Improved {total_stats['layers_improved']} layers")
        
        return total_stats
    
    def _finetune_layer(self,
                       layer: nn.Module,
                       quantizer: Union[AdaRoundQuantizer, AdaQuantQuantizer],
                       calibration_data: torch.Tensor,
                       validation_data: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Fine-tune a single layer."""
        import time
        start_time = time.time()
        
        # Get original layer output
        layer.eval()
        with torch.no_grad():
            original_output = layer(calibration_data)
            
        # Define layer forward function
        def layer_forward_fn(weights, inputs=None):
            if inputs is None:
                inputs = calibration_data
            if isinstance(layer, nn.Linear):
                return F.linear(inputs, weights, layer.bias)
            elif isinstance(layer, nn.Conv2d):
                return F.conv2d(inputs, weights, layer.bias, 
                              layer.stride, layer.padding)
        
        # Fine-tune based on method
        if self.method == 'adaround':
            stats = quantizer.train_rounding(
                layer.weight.data, original_output, 
                lambda w: layer_forward_fn(w)
            )
        else:  # adaquant
            stats = quantizer.optimize_parameters(
                layer.weight.data, calibration_data, original_output,
                layer_forward_fn
            )
        
        # Check if improvement occurred (selective update)
        improved = False
        if hasattr(self.config, 'selective_update') and self.config.selective_update:
            if validation_data is not None:
                improved = self._validate_improvement(
                    layer, quantizer, validation_data
                )
            else:
                # Use training loss as proxy
                improved = stats['final_loss'] < stats.get('initial_loss', float('inf'))
        else:
            improved = True  # Always accept changes
            
        stats.update({
            'improved': improved,
            'processing_time': time.time() - start_time
        })
        
        return stats
    
    def _validate_improvement(self,
                            layer: nn.Module,
                            quantizer: Union[AdaRoundQuantizer, AdaQuantQuantizer],
                            validation_data: torch.Tensor) -> bool:
        """Validate if quantization improved the layer."""
        # This is a simplified validation - in practice, you'd want
        # to measure end-to-end model accuracy
        layer.eval()
        with torch.no_grad():
            original_output = layer(validation_data)
            
            # Simulate quantized layer
            if isinstance(quantizer, AdaRoundQuantizer):
                quantized_weights = quantizer._hard_quantize(layer.weight.data)
            else:
                quantized_weights = quantizer.quantize(layer.weight.data)
                
            dequantized_weights = quantizer.dequantize(quantized_weights)
            
            if isinstance(layer, nn.Linear):
                quantized_output = F.linear(validation_data, dequantized_weights, layer.bias)
            else:
                quantized_output = F.conv2d(validation_data, dequantized_weights, layer.bias,
                                          layer.stride, layer.padding)
            
            # Compare outputs
            mse = F.mse_loss(quantized_output, original_output)
            
            # Simple threshold - in practice, you'd use more sophisticated metrics
            return mse.item() < 0.1  # Threshold for acceptable error
