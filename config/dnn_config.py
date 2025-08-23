"""DNN-specific configuration class for vertex time prediction."""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict
from .base_config import BaseConfig


@dataclass
class DNNConfig(BaseConfig):
    """Configuration for two-stage DNN model."""
    
    # Model architecture type
    model_architecture: str = "two_stage_dnn"
    
    # Cell-level processing parameters
    cell_encoder_units: List[int] = None
    cell_dropout_rate: float = 0.2
    cell_activation: str = 'relu'
    
    # Attention pooling parameters
    use_attention_pooling: bool = True
    attention_pooling_masked: bool = True
    attention_hidden_units: int = 32
    
    # Event-level processing parameters  
    event_encoder_units: List[int] = None
    event_dropout_rates: List[float] = None
    use_batch_norm: bool = True
    
    # Training parameters
    learning_rate: float = 1e-4
    lr_reduction_factor: float = 0.5
    
    # Model name override
    model_name: str = "dnn_model"
    
    def __post_init__(self):
        """Initialize additional parameters."""
        super().__post_init__()
        
        if self.cell_encoder_units is None:
            self.cell_encoder_units = [64, 32]
            
        if self.event_encoder_units is None:
            self.event_encoder_units = [256, 128, 64]
            
        if self.event_dropout_rates is None:
            self.event_dropout_rates = [0.3, 0.2, 0.1]
    
    def validate_config(self):
        """Validate DNN-specific configuration."""
        # Cell encoder validations
        assert len(self.cell_encoder_units) > 0, "cell_encoder_units cannot be empty"
        assert all(units > 0 for units in self.cell_encoder_units), "All cell encoder units must be positive"
        assert 0 <= self.cell_dropout_rate < 1, "cell_dropout_rate must be between 0 and 1"
        
        # Event encoder validations  
        assert len(self.event_encoder_units) > 0, "event_encoder_units cannot be empty"
        assert all(units > 0 for units in self.event_encoder_units), "All event encoder units must be positive"
        assert len(self.event_encoder_units) == len(self.event_dropout_rates), \
            "event_encoder_units and event_dropout_rates must have same length"
        assert all(0 <= rate < 1 for rate in self.event_dropout_rates), \
            "All dropout rates must be between 0 and 1"
        
        # Attention parameters
        assert self.attention_hidden_units > 0, "attention_hidden_units must be positive"
        
        # Call parent validation
        super().validate_config()
    
    def print_config(self):
        """Print DNN-specific configuration parameters."""
        # Call parent print_config first
        super().print_config()
        
        # Add DNN-specific parameters
        print(f"\nDNN Architecture Parameters:")
        print(f"  model_architecture: {self.model_architecture}")
        print(f"  cell_encoder_units: {self.cell_encoder_units}")
        print(f"  cell_dropout_rate: {self.cell_dropout_rate}")
        print(f"  cell_activation: {self.cell_activation}")
        print(f"  use_attention_pooling: {self.use_attention_pooling}")
        print(f"  attention_pooling_masked: {self.attention_pooling_masked}")
        print(f"  attention_hidden_units: {self.attention_hidden_units}")
        print(f"  event_encoder_units: {self.event_encoder_units}")
        print(f"  event_dropout_rates: {self.event_dropout_rates}")
        print(f"  use_batch_norm: {self.use_batch_norm}")
