"""Base configuration class for all models with YAML support."""

import os
from dataclasses import dataclass, fields
from typing import List, Optional, Dict, Any, Union
import json
import yaml


def get_external_dir(dir_name: str, default_relative_path: str) -> str:
    """
    Get external directory path with environment variable support.
    
    Args:
        dir_name: Name of the directory (e.g., 'models', 'results')
        default_relative_path: Default path relative to project root
        
    Returns:
        Absolute path to the directory
    """
    # Try environment variable first
    env_var = f"VERTEX_TIME_{dir_name.upper()}_DIR"
    env_path = os.environ.get(env_var)
    
    if env_path:
        return os.path.abspath(env_path)
    
    # Use default external path (one level up from project)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    external_path = os.path.join(os.path.dirname(project_root), dir_name)
    
    return external_path


@dataclass
class BaseConfig:
    """Base configuration class containing common parameters."""
    
    # Data parameters
    data_dir: str = "../selected_h5/"
    num_files: int = 50
    max_cells: int = 40
    min_cells: int = 3
    cell_selection_feature: str = 'Cell_e'
    
    # Feature selection parameters
    use_spatial_features: bool = False
    use_track_features: bool = True   # NEW: Include track matching features
    use_jet_features: bool = False    # NEW: Include jet matching features
    
    # Cell filtering parameters
    require_valid_cells: bool = True
    use_cell_track_matching: bool = True
    use_cell_jet_matching: bool = False  # NEW: Enable cell-jet matching filter
    additional_cell_filters: Dict[str, Any] = None
    
    # Data split parameters
    test_size: float = 0.3
    val_split: float = 1/3  # Fraction of test_size for validation
    random_state: int = 42
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 15
    lr_patience: int = 5
    min_lr: float = 1e-7
    
    # Model save parameters - Updated to use external directories
    models_base_dir: str = None  # Will be set in __post_init__
    model_name: str = "base_model"
    
    # Feature definitions
    spatial_features: List[str] = None
    vertex_spatial_features: List[str] = None
    all_cell_features: List[str] = None
    track_features: List[str] = None      # NEW: Track matching features
    jet_features: List[str] = None        # NEW: Jet matching features
    skip_normalization: List[str] = None
    
    def __post_init__(self):
        """Initialize feature lists and paths after dataclass initialization."""
        # Set default external models directory if not specified
        if self.models_base_dir is None:
            self.models_base_dir = get_external_dir("models", "models")
        
        # Ensure models directory exists
        os.makedirs(self.models_base_dir, exist_ok=True)
        
        if self.spatial_features is None:
            self.spatial_features = ['Cell_x', 'Cell_y', 'Cell_z']
            
        if self.vertex_spatial_features is None:
            self.vertex_spatial_features = ["HSvertex_reco_x", "HSvertex_reco_y", "HSvertex_reco_z"]
            
        if self.all_cell_features is None:
            # Base cell features (physical properties only)
            self.all_cell_features = [
                'Cell_x', 'Cell_y', 'Cell_z', 'Cell_eta', 'Cell_phi', 'Cell_Barrel', 'Cell_layer',
                'Cell_time_TOF_corrected', 'Cell_e', 'Cell_significance'
            ]
            
        # NEW: Initialize track and jet features separately
        if self.track_features is None:
            self.track_features = [
                'matched_track_pt',
                'matched_track_deltaR'
            ]
            
        if self.jet_features is None:
            self.jet_features = [
                'matched_jet_pt',
                'matched_jet_eta', 
                'matched_jet_phi',
                'matched_jet_width',
                'matched_jet_deltaR'
            ]
    
        if self.skip_normalization is None:
            self.skip_normalization = ['Cell_time_TOF_corrected', 'Cell_Barrel', 'Cell_layer']
            
        if self.additional_cell_filters is None:
            self.additional_cell_filters = {}
    
    @property
    def cell_features(self) -> List[str]:
        """Get cell features based on feature selection settings."""
        # Start with base cell features
        features = self.all_cell_features.copy()
        
        # Add spatial features if enabled
        if self.use_spatial_features:
            # spatial features are already in all_cell_features, so no change needed
            pass
        else:
            # Remove spatial features if disabled
            features = [f for f in features if f not in self.spatial_features]
        
        # Add track features if enabled
        if self.use_track_features:
            for track_feature in self.track_features:
                if track_feature not in features:
                    features.append(track_feature)
        
        # Add jet features if enabled
        if self.use_jet_features:
            for jet_feature in self.jet_features:
                if jet_feature not in features:
                    features.append(jet_feature)
        
        return features
    
    @property
    def model_dir(self) -> str:
        """Get model directory path."""
        return os.path.join(self.models_base_dir, self.model_name)
    
    @property
    def model_path(self) -> str:
        """Get model file path."""
        return os.path.join(self.model_dir, "model.keras")
    
    @property
    def plots_dir(self) -> str:
        """Get evaluation plots directory."""
        return os.path.join(self.model_dir, "evaluation_plots")
    
    def get_cell_filtering_description(self) -> str:
        """Get description of cell filtering conditions."""
        conditions = []
        
        if self.require_valid_cells:
            conditions.append("valid == True")
            
        if self.use_cell_track_matching:
            conditions.append("matched_track_HS == 1")
            
        # NEW: Add jet matching description
        if self.use_cell_jet_matching:
            conditions.append("cell_jet_matched == True")
            
        for key, value in self.additional_cell_filters.items():
            conditions.append(f"{key} == {value}")
            
        return " & ".join(conditions) if conditions else "No filtering"
    
    def create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def save_config(self):
        """Save configuration to JSON file."""
        self.create_directories()
        config_path = os.path.join(self.model_dir, "config.json")
        
        # Convert to dictionary, handling non-serializable types
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def save_yaml(self, filepath: str = None):
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save YAML file. If None, saves to model directory.
        """
        if filepath is None:
            self.create_directories()
            filepath = os.path.join(self.model_dir, "config.yaml")
        
        # Convert to dictionary
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Configuration instance
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Get valid field names for this class
        valid_fields = {field.name for field in fields(cls)}
        
        # Filter yaml_data to only include valid fields
        filtered_data = {k: v for k, v in yaml_data.items() if k in valid_fields}
        
        # Create instance with filtered data
        instance = cls(**filtered_data)
        
        print(f"Configuration loaded from: {yaml_path}")
        print(f"Model name: {instance.model_name}")
        print(f"Models directory: {instance.models_base_dir}")
        print(f"Cell filtering: {instance.get_cell_filtering_description()}")
        
        return instance
    
    @classmethod
    def load_config(cls, model_dir: str):
        """Load configuration from JSON file."""
        config_path = os.path.join(model_dir, "config.json")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update configuration parameters from dictionary.
        
        Args:
            updates: Dictionary of parameter updates
        """
        valid_fields = {field.name for field in fields(self)}
        
        for key, value in updates.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")
    
    def update_from_args(self, args):
        """
        Update configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Map of argument names to config attribute names
        arg_mapping = {
            'data_dir': 'data_dir',
            'model_name': 'model_name',
            'epochs': 'epochs',
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'max_cells': 'max_cells',
            'min_cells': 'min_cells',
            'use_spatial': 'use_spatial_features'
        }
        
        for arg_name, config_attr in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                # Special handling for use_spatial flag
                if arg_name == 'use_spatial':
                    # Only set to True if the flag is explicitly provided
                    # If not provided, keep the YAML value
                    if getattr(args, arg_name):
                        setattr(self, config_attr, True)
                else:
                    setattr(self, config_attr, getattr(args, arg_name))
    
    def print_config(self):
        """Print configuration parameters in a formatted way."""
        print("=" * 60)
        print(f"CONFIGURATION: {self.model_name}")
        print("=" * 60)
        
        # Group parameters by category
        categories = {
            "Data Parameters": [
                'data_dir', 'num_files', 'max_cells', 'min_cells', 
                'cell_selection_feature'
            ],
            "Feature Selection Parameters": [
                'use_spatial_features', 'use_track_features', 'use_jet_features'
            ],
            "Cell Filtering Parameters": [
                'require_valid_cells', 'use_cell_track_matching', 'use_cell_jet_matching', 'additional_cell_filters'
            ],
            "Training Parameters": [
                'batch_size', 'epochs', 'early_stopping_patience', 
                'lr_patience', 'min_lr'
            ],
            "Model Save Parameters": [
                'models_base_dir', 'model_name'
            ]
        }
        
        for category, params in categories.items():
            print(f"\n{category}:")
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    if param == 'additional_cell_filters':
                        print(f"  {param}: {dict(value) if value else '{}'}")
                    else:
                        print(f"  {param}: {value}")
        
        # Add special filtering description
        print(f"\nCell Filtering Description:")
        print(f"  Conditions: {self.get_cell_filtering_description()}")
        
        # Add path information
        print(f"\nPath Information:")
        print(f"  Model directory: {self.model_dir}")
        print(f"  Model file: {self.model_path}")
        print(f"  Plots directory: {self.plots_dir}")
        
        # Add feature information
        print(f"\nFeature Information:")
        print(f"  Total cell features: {len(self.cell_features)}")
        print(f"  Base cell features: {self.all_cell_features}")
        if self.use_track_features:
            print(f"  Track features: {self.track_features}")
        if self.use_jet_features:
            print(f"  Jet features: {self.jet_features}")
        print(f"  Final cell features: {self.cell_features}")
        
        # Add architecture parameters if they exist
        arch_params = ['d_model', 'num_heads', 'dff', 'num_transformer_blocks', 'dropout_rate']
        if any(hasattr(self, param) for param in arch_params):
            print(f"\nArchitecture Parameters:")
            for param in arch_params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    print(f"  {param}: {value}")
        
        print("=" * 60)
    
    def validate_config(self):
        """Validate configuration parameters."""
        # Basic validations
        assert self.max_cells > 0, "max_cells must be positive"
        assert self.min_cells > 0, "min_cells must be positive"
        assert self.min_cells <= self.max_cells, "min_cells must be <= max_cells"
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        
        # Feature validations
        assert len(self.all_cell_features) > 0, "all_cell_features cannot be empty"
        assert len(self.cell_features) > 0, "No cell features available with current settings"
        
        # Path validations
        assert os.path.isabs(self.models_base_dir), "models_base_dir should be absolute path"
        
        # NEW: Feature selection validations
        if self.use_track_features and not self.track_features:
            print("Warning: use_track_features is enabled but no track_features defined")
        
        if self.use_jet_features and not self.jet_features:
            print("Warning: use_jet_features is enabled but no jet_features defined")
        
        # NEW: Track feature validations
        if self.use_track_features:
            available_track_features = ['matched_track_pt', 'matched_track_deltaR', 'matched_track_HS']
            missing_track_features = []
            for track_feature in self.track_features:
                if track_feature not in available_track_features:
                    missing_track_features.append(track_feature)
            
            if missing_track_features:
                print(f"Warning: Unknown track features specified: {missing_track_features}")
                print(f"Available track features: {available_track_features}")
        
        # NEW: Jet feature validations
        if self.use_jet_features:
            # Validate that jet features are available in data
            available_jet_features = [
                'cell_jet_matched', 'matched_jet_pt', 'matched_jet_eta', 
                'matched_jet_phi', 'matched_jet_width', 'matched_jet_deltaR'
            ]
            missing_jet_features = []
            for jet_feature in self.jet_features:
                if jet_feature not in available_jet_features:
                    missing_jet_features.append(jet_feature)
            
            if missing_jet_features:
                print(f"Warning: Unknown jet features specified: {missing_jet_features}")
                print(f"Available jet features: {available_jet_features}")
        
        # Cell filtering validations
        if not self.require_valid_cells and not self.use_cell_track_matching and not self.use_cell_jet_matching and not self.additional_cell_filters:
            print("Warning: No cell filtering enabled. This may include low-quality cells.")
        
        # NEW: Matching filter validation
        if self.use_cell_track_matching:
            print("Note: Cell-track matching filter enabled. Ensure your data contains 'matched_track_HS' field.")
        
        if self.use_cell_jet_matching:
            print("Note: Cell-jet matching filter enabled. Ensure your data contains 'cell_jet_matched' field.")
        
        # Check if additional_cell_filters contains valid keys
        if self.additional_cell_filters:
            all_available_features = self.all_cell_features + self.track_features + self.jet_features
            for filter_key in self.additional_cell_filters.keys():
                if filter_key not in all_available_features:
                    print(f"Warning: Filter key '{filter_key}' not in available features")
        
        print("Configuration validation passed.")
        print(f"Using {len(self.cell_features)} out of {len(self.all_cell_features)} available features")
        print(f"Cell filtering: {self.get_cell_filtering_description()}")
        print(f"External models directory: {self.models_base_dir}")
