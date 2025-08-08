"""Transformer-specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for Transformer model."""
    
    # Model architecture parameters
    d_model: int = 128
    num_heads: int = 8
    dff: int = 256
    num_transformer_blocks: int = 3
    dropout_rate: float = 0.1
    
    # Dense layer parameters
    vertex_dense_units: int = 64
    final_dense_units: list = None
    final_dropout_rates: list = None
    use_batch_norm: bool = True
    
    # Training parameters
    learning_rate: float = 1e-4
    lr_reduction_factor: float = 0.5
    
    # Model name override
    model_name: str = "transformer_model"
    
    def __post_init__(self):
        """Initialize additional parameters."""
        super().__post_init__()
        
        if self.final_dense_units is None:
            self.final_dense_units = [256, 128, 64]
            
        if self.final_dropout_rates is None:
            self.final_dropout_rates = [0.3, 0.2, 0.1]
    
    @property
    def max_position(self) -> int:
        """Calculate maximum position for positional encoding."""
        return self.max_cells * 2  # Allow for larger sequences than expected
    
    def validate_config(self):
        """Validate transformer-specific configuration."""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert len(self.final_dense_units) == len(self.final_dropout_rates), \
            "final_dense_units and final_dropout_rates must have same length"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.num_transformer_blocks > 0, "num_transformer_blocks must be positive"
