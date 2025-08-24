"""Improved data processor with physics-informed features."""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import logging

from .data_processor import DataProcessor
from .physics_features import PhysicsFeatureEngineer

logger = logging.getLogger(__name__)


class ImprovedDataProcessor(DataProcessor):
    """Enhanced data processor with physics-informed features."""
    
    def __init__(self, config):
        """Initialize improved data processor."""
        super().__init__(config)
        
        # Initialize physics feature engineer
        self.physics_engineer = PhysicsFeatureEngineer(config)
        self.use_physics_features = getattr(config, 'use_physics_informed_features', False)
        
        # Enhanced feature scaling
        self.time_scaler = None
        self.feature_scalers = {}
        
    def create_datasets_with_physics(
        self, 
        cell_sequences: List[List[List[float]]], 
        vertex_times: np.ndarray, 
        vertex_features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Create train/val/test datasets with physics-informed features.
        
        Args:
            cell_sequences: Original cell sequences
            vertex_times: Target vertex times
            vertex_features: Vertex feature array
            feature_names: Original feature names
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        # Add physics-informed features if enabled
        if self.use_physics_features:
            print("Adding physics-informed features...")
            cell_sequences, feature_names = self.physics_engineer.add_physics_features(
                cell_sequences, feature_names
            )
            
            # Compute event-level features
            event_features = self.physics_engineer.compute_event_level_features(
                cell_sequences, feature_names
            )
            print(f"Added event-level features: {event_features.shape}")
        else:
            event_features = np.empty((len(cell_sequences), 0))
        
        # Apply enhanced preprocessing
        cell_sequences = self._apply_enhanced_preprocessing(cell_sequences, feature_names)
        
        # Create padded sequences and masks
        max_seq_len = min(max(len(seq) for seq in cell_sequences), self.config.max_cells)
        
        # Pad sequences
        feature_dim = len(cell_sequences[0][0]) if cell_sequences and cell_sequences[0] else len(feature_names)
        padded_sequences = self.apply_smart_padding(cell_sequences, max_seq_len, feature_dim)
        
        # Create attention masks
        attention_masks = self.create_attention_mask(cell_sequences, max_seq_len)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        indices = np.arange(len(padded_sequences))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Second split: separate train and validation
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=self.config.val_split, random_state=self.config.random_state
        )
        
        # Create data dictionaries
        def create_data_dict(indices):
            return {
                'cell_sequences': padded_sequences[indices],
                'vertex_times': vertex_times[indices],
                'vertex_features': vertex_features[indices],
                'attention_masks': attention_masks[indices],
                'event_features': event_features[indices] if event_features.shape[1] > 0 else None
            }
        
        train_data = create_data_dict(train_idx)
        val_data = create_data_dict(val_idx)
        test_data = create_data_dict(test_idx)
        
        print(f"Dataset sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print(f"Feature dimension: {feature_dim}")
        print(f"Max sequence length: {max_seq_len}")
        
        return train_data, val_data, test_data
    
    def _apply_enhanced_preprocessing(self, cell_sequences: List, feature_names: List[str]) -> List:
        """Apply enhanced preprocessing including selective normalization."""
        if not cell_sequences:
            return cell_sequences
        
        # Find feature indices
        feature_indices = {name: i for i, name in enumerate(feature_names)}
        
        # Features that should be normalized
        normalize_features = [name for name in feature_names if name not in self.config.skip_normalization]
        
        # Special handling for time features if requested
        if getattr(self.config, 'time_calibration_aware', False):
            time_features = [name for name in feature_names if 'time' in name.lower()]
            for time_feat in time_features:
                if time_feat in normalize_features:
                    self._apply_time_specific_normalization(cell_sequences, feature_indices[time_feat])
        
        # Apply standard normalization to other features
        for feature_name in normalize_features:
            if 'time' not in feature_name.lower() or not getattr(self.config, 'time_calibration_aware', False):
                feature_idx = feature_indices[feature_name]
                self._normalize_feature(cell_sequences, feature_idx, feature_name)
        
        return cell_sequences
    
    def _apply_time_specific_normalization(self, cell_sequences: List, time_idx: int):
        """Apply specialized normalization for time features."""
        # Collect all time values
        all_times = []
        for sequence in cell_sequences:
            for cell in sequence:
                if len(cell) > time_idx:
                    all_times.append(cell[time_idx])
        
        if not all_times:
            return
        
        all_times = np.array(all_times)
        
        # Use robust statistics for time normalization
        time_median = np.median(all_times)
        time_mad = np.median(np.abs(all_times - time_median))  # Median Absolute Deviation
        
        # Robust scaling: (x - median) / (1.4826 * MAD)
        # 1.4826 makes MAD approximately equal to std for normal distribution
        time_scale = 1.4826 * time_mad if time_mad > 0 else 1.0
        
        # Apply normalization
        for sequence in cell_sequences:
            for cell in sequence:
                if len(cell) > time_idx:
                    cell[time_idx] = (cell[time_idx] - time_median) / time_scale
        
        logger.info(f"Applied robust time normalization: median={time_median:.2f}, scale={time_scale:.2f}")
    
    def _normalize_feature(self, cell_sequences: List, feature_idx: int, feature_name: str):
        """Apply standard normalization to a specific feature."""
        # Collect all values for this feature
        all_values = []
        for sequence in cell_sequences:
            for cell in sequence:
                if len(cell) > feature_idx:
                    all_values.append(cell[feature_idx])
        
        if not all_values:
            return
        
        all_values = np.array(all_values).reshape(-1, 1)
        
        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(all_values)
        
        # Store scaler for later use
        self.feature_scalers[feature_name] = scaler
        
        # Apply normalization
        for sequence in cell_sequences:
            for cell in sequence:
                if len(cell) > feature_idx:
                    normalized = scaler.transform([[cell[feature_idx]]])
                    cell[feature_idx] = normalized[0, 0]
    
    def create_tensorflow_datasets(
        self, 
        train_data: Dict, 
        val_data: Dict, 
        test_data: Dict,
        batch_size: int = None
    ) -> Tuple[Any, Any, Any]:
        """
        Create TensorFlow datasets from processed data.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary  
            test_data: Test data dictionary
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        import tensorflow as tf
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        def create_tf_dataset(data_dict, is_training=False):
            # Base inputs
            inputs = {
                'cell_sequence': data_dict['cell_sequences'],
                'vertex_features': data_dict['vertex_features'], 
                'attention_mask': data_dict['attention_masks']
            }
            
            # Add event features if available
            if data_dict['event_features'] is not None:
                inputs['event_features'] = data_dict['event_features']
            
            targets = data_dict['vertex_times']
            
            # Create dataset
            if data_dict['event_features'] is not None:
                dataset = tf.data.Dataset.from_tensor_slices((
                    (inputs['cell_sequence'], inputs['vertex_features'], 
                     inputs['attention_mask'], inputs['event_features']),
                    targets
                ))
            else:
                dataset = tf.data.Dataset.from_tensor_slices((
                    (inputs['cell_sequence'], inputs['vertex_features'], inputs['attention_mask']),
                    targets
                ))
            
            # Apply batching and shuffling
            if is_training:
                dataset = dataset.shuffle(buffer_size=10000, seed=self.config.random_state)
            
            dataset = dataset.batch(batch_size)
            
            # Prefetch for performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        train_dataset = create_tf_dataset(train_data, is_training=True)
        val_dataset = create_tf_dataset(val_data, is_training=False)
        test_dataset = create_tf_dataset(test_data, is_training=False)
        
        return train_dataset, val_dataset, test_dataset