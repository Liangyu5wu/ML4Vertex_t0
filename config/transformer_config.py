"""Transformer-specific configuration."""

from dataclasses import dataclass
from typing import List, Optional
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
    
    # Detector calibration parameters
    use_detector_params: bool = False
    calibration_data_file: str = "HStrackmatching_calibration.txt"
    
    # Calibration validation parameters
    calibration_validation: bool = False
    validation_detector_type: int = 1  # 1=barrel, 0=endcap
    validation_layer: int = 1  # 1, 2, or 3
    gaussian_fit_range: float = 120  # Range for Gaussian fitting in baseline checks
    
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
    
    def validate_detector_params(self):
        """Validate detector calibration parameters."""
        if not self.use_detector_params:
            return  # No validation needed if detector params are disabled
        
        detector_param_names = [
            'emb1_params', 'emb2_params', 'emb3_params',
            'eme1_params', 'eme2_params', 'eme3_params'
        ]
        
        missing_params = []
        invalid_length_params = []
        
        for param_name in detector_param_names:
            param_value = getattr(self, param_name)
            
            if param_value is None:
                missing_params.append(param_name)
            elif not isinstance(param_value, (list, tuple)):
                raise ValueError(f"{param_name} must be a list or tuple, got {type(param_value)}")
            elif len(param_value) != 7:
                invalid_length_params.append(f"{param_name} (length: {len(param_value)})")
        
        if missing_params:
            raise ValueError(
                f"use_detector_params=True requires all detector parameters to be specified. "
                f"Missing: {', '.join(missing_params)}"
            )
        
        if invalid_length_params:
            raise ValueError(
                f"All detector parameter arrays must have exactly 7 elements. "
                f"Invalid lengths: {', '.join(invalid_length_params)}"
            )
        
        # Validate that Cell_Barrel and Cell_layer are in cell features
        if 'Cell_Barrel' not in self.all_cell_features:
            raise ValueError("Cell_Barrel must be in all_cell_features when using detector params")
        
        if 'Cell_layer' not in self.all_cell_features:
            raise ValueError("Cell_layer must be in all_cell_features when using detector params")
        
        print("Detector parameter validation passed.")
        print(f"Using detector calibration with 6 parameter sets of 7 values each")
    
    def validate_config(self):
        """Validate transformer-specific configuration."""
        # Basic transformer validations
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert len(self.final_dense_units) == len(self.final_dropout_rates), \
            "final_dense_units and final_dropout_rates must have same length"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.num_transformer_blocks > 0, "num_transformer_blocks must be positive"
        
        # Validate detector parameters if enabled
        self.validate_detector_params()
        
        # Call parent validation
        super().validate_config()
    
    def get_detector_params_description(self) -> str:
        """Get description of detector parameter configuration."""
        if not self.use_detector_params:
            return "Detector parameters: Disabled"
        
        param_info = []
        detector_params = {
            'EMB1': self.emb1_params,
            'EMB2': self.emb2_params, 
            'EMB3': self.emb3_params,
            'EME1': self.eme1_params,
            'EME2': self.eme2_params,
            'EME3': self.eme3_params
        }
        
        for name, params in detector_params.items():
            if params is not None:
                param_info.append(f"{name}: [{params[0]:.2f}, ..., {params[-1]:.2f}]")
        
        return f"Detector parameters: Enabled\n  " + "\n  ".join(param_info)
    
    def print_config(self):
        """Print configuration parameters in a formatted way."""
        # Call parent print_config first
        super().print_config()
        
        # Add detector parameter information
        if hasattr(self, 'use_detector_params'):
            print(f"\nDetector Calibration Parameters:")
            print(f"  {self.get_detector_params_description()}")
