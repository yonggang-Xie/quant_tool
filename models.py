import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLPClassifier(nn.Module):
    """
    Simple Multi-Layer Perceptron for classification tasks.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int, 
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Calculate the size of flattened features
        # Assuming input size is 32x32, after 3 maxpool operations: 32/8 = 4
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet-like architectures.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class SimpleResNet(nn.Module):
    """
    Simplified ResNet architecture for testing quantization.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Simple transformer block for sequence modeling.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class SimpleTransformer(nn.Module):
    """
    Simple transformer model for sequence classification.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, ff_dim: int, max_seq_len: int, 
                 num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(x) * (self.embed_dim ** 0.5)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer blocks
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=0)  # (batch_size, embed_dim)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_sample_models() -> dict:
    """
    Create a dictionary of sample models for testing quantization.
    
    Returns:
        Dictionary containing various model architectures
    """
    models = {
        'mlp_small': SimpleMLPClassifier(784, [128, 64], 10),
        'mlp_large': SimpleMLPClassifier(784, [512, 256, 128], 10),
        'cnn_simple': SimpleCNN(3, 10),
        'resnet_simple': SimpleResNet(3, 10),
        'transformer_small': SimpleTransformer(
            vocab_size=1000, embed_dim=128, num_heads=4, 
            num_layers=2, ff_dim=256, max_seq_len=100, num_classes=10
        )
    }
    
    return models


def initialize_model_weights(model: nn.Module, init_type: str = 'xavier') -> None:
    """
    Initialize model weights using specified initialization scheme.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization ('xavier', 'kaiming', 'normal')
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(module.weight, 0, 0.02)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def generate_sample_data(model_type: str, batch_size: int = 32) -> torch.Tensor:
    """
    Generate sample input data for different model types.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'transformer')
        batch_size: Batch size for generated data
        
    Returns:
        Sample input tensor
    """
    if model_type == 'mlp':
        return torch.randn(batch_size, 784)
    elif model_type == 'cnn':
        return torch.randn(batch_size, 3, 32, 32)
    elif model_type == 'transformer':
        return torch.randint(0, 1000, (batch_size, 100))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class ModelFactory:
    """
    Factory class for creating and managing different model architectures.
    """
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> nn.Module:
        """
        Create a model by name with optional parameters.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional parameters for model creation
            
        Returns:
            Created PyTorch model
        """
        if model_name == 'mlp':
            input_size = kwargs.get('input_size', 784)
            hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
            num_classes = kwargs.get('num_classes', 10)
            dropout_rate = kwargs.get('dropout_rate', 0.2)
            return SimpleMLPClassifier(input_size, hidden_sizes, num_classes, dropout_rate)
        
        elif model_name == 'cnn':
            input_channels = kwargs.get('input_channels', 3)
            num_classes = kwargs.get('num_classes', 10)
            return SimpleCNN(input_channels, num_classes)
        
        elif model_name == 'resnet':
            input_channels = kwargs.get('input_channels', 3)
            num_classes = kwargs.get('num_classes', 10)
            return SimpleResNet(input_channels, num_classes)
        
        elif model_name == 'transformer':
            vocab_size = kwargs.get('vocab_size', 1000)
            embed_dim = kwargs.get('embed_dim', 128)
            num_heads = kwargs.get('num_heads', 4)
            num_layers = kwargs.get('num_layers', 2)
            ff_dim = kwargs.get('ff_dim', 256)
            max_seq_len = kwargs.get('max_seq_len', 100)
            num_classes = kwargs.get('num_classes', 10)
            dropout = kwargs.get('dropout', 0.1)
            return SimpleTransformer(vocab_size, embed_dim, num_heads, num_layers,
                                   ff_dim, max_seq_len, num_classes, dropout)
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """
        Get information about a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        layer_types = {}
        for module in model.modules():
            module_type = type(module).__name__
            layer_types[module_type] = layer_types.get(module_type, 0) + 1
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_types': layer_types,
            'model_class': type(model).__name__
        }
