"""Data loading utilities for HDF5 files with configurable cell filtering."""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
from config.base_config import BaseConfig


class DataLoader:
    """Load and preprocess data from HDF5 files with configurable cell filtering."""
    
    def __init__(self, config: BaseConfig):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object containing data parameters
        """
        self.config = config
        
    def get_file_paths(self) -> List[str]:
        """Get list of HDF5 file paths."""
        return [
            os.path.join(self.config.data_dir, f"output_{i:03d}.h5") 
            for i in range(self.config.num_files)
        ]
    
    def apply_cell_filtering(self, event_cells: np.ndarray) -> np.ndarray:
        """
        Apply configurable cell filtering based on configuration.
        
        Args:
            event_cells: Array of cells for a single event
            
        Returns:
            Filtered array of cells
        """
        # Start with all cells
        mask = np.ones(len(event_cells), dtype=bool)
        
        # Apply valid cell filter
        if self.config.require_valid_cells:
            valid_mask = event_cells['valid'] == True
            mask = mask & valid_mask
        
        # Apply cell-track matching filter
        if self.config.use_cell_track_matching:
            track_matching_mask = event_cells['matched_track_HS'] == 1
            mask = mask & track_matching_mask
        
        # Apply additional custom filters
        if self.config.additional_cell_filters:
            for filter_key, filter_value in self.config.additional_cell_filters.items():
                if filter_key in event_cells.dtype.names:
                    additional_mask = event_cells[filter_key] == filter_value
                    mask = mask & additional_mask
                else:
                    print(f"Warning: Filter key '{filter_key}' not found in cell data. Skipping this filter.")
        
        return event_cells[mask]
    
    def get_filtering_statistics(self, event_cells: np.ndarray) -> dict:
        """
        Get statistics about cell filtering for debugging/monitoring.
        
        Args:
            event_cells: Original array of cells
            
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            'total_cells': len(event_cells),
            'valid_cells': 0,
            'track_matched_cells': 0,
            'final_filtered_cells': 0
        }
        
        if len(event_cells) == 0:
            return stats
        
        # Count valid cells
        if 'valid' in event_cells.dtype.names:
            stats['valid_cells'] = np.sum(event_cells['valid'] == True)
        
        # Count track-matched cells
        if 'matched_track_HS' in event_cells.dtype.names:
            stats['track_matched_cells'] = np.sum(event_cells['matched_track_HS'] == 1)
        
        # Count cells after all filtering
        filtered_cells = self.apply_cell_filtering(event_cells)
        stats['final_filtered_cells'] = len(filtered_cells)
        
        return stats
    
    def load_data_from_files(
        self, 
        file_paths: Optional[List[str]] = None,
        print_filtering_stats: bool = True
    ) -> Tuple[List[List[List[float]]], np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from HDF5 files with configurable cell filtering.
        
        Args:
            file_paths: List of file paths to load. If None, uses default paths.
            print_filtering_stats: Whether to print cell filtering statistics
            
        Returns:
            Tuple of (cell_sequences, vertex_features, vertex_times, sequence_lengths)
        """
        if file_paths is None:
            file_paths = self.get_file_paths()
            
        all_cell_sequences = []
        all_vertex_features = []
        all_vertex_times = []
        sequence_lengths = []
        
        # Statistics for monitoring filtering effectiveness
        filtering_stats = {
            'total_events': 0,
            'events_with_cells': 0,
            'events_after_min_cells_filter': 0,
            'total_cells_before_filtering': 0,
            'total_cells_after_filtering': 0,
            'cells_removed_by_valid_filter': 0,
            'cells_removed_by_track_filter': 0,
            'cells_removed_by_additional_filters': 0
        }
        
        print(f"Cell filtering configuration:")
        print(f"  Require valid cells: {self.config.require_valid_cells}")
        print(f"  Use cell-track matching: {self.config.use_cell_track_matching}")
        print(f"  Additional filters: {self.config.additional_cell_filters}")
        print(f"  Filtering description: {self.config.get_cell_filtering_description()}")
        print(f"Using spatial features: {self.config.use_spatial_features}")
        print(f"Cell features used: {self.config.cell_features}")
        print(f"Min cells required: {self.config.min_cells}, Max cells considered: {self.config.max_cells}")
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            print(f"Processing {file_path}...")
            try:
                cell_seq, vertex_feat, vertex_time, seq_len, file_stats = self._process_file(file_path)
                all_cell_sequences.extend(cell_seq)
                all_vertex_features.extend(vertex_feat)
                all_vertex_times.extend(vertex_time)
                sequence_lengths.extend(seq_len)
                
                # Accumulate statistics
                for key in filtering_stats:
                    filtering_stats[key] += file_stats.get(key, 0)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        sequence_lengths = np.array(sequence_lengths)
        print(f"Processed {len(all_vertex_times)} valid events")
        
        # Print filtering statistics
        if print_filtering_stats and filtering_stats['total_events'] > 0:
            self._print_filtering_statistics(filtering_stats)
        
        self._print_sequence_statistics(sequence_lengths)
        
        return (all_cell_sequences, np.array(all_vertex_features), 
                np.array(all_vertex_times), sequence_lengths)
    
    def _process_file(self, file_path: str) -> Tuple[List, List, List, List, dict]:
        """Process a single HDF5 file with detailed statistics."""
        cell_sequences = []
        vertex_features = []
        vertex_times = []
        sequence_lengths = []
        
        # File-level statistics
        file_stats = {
            'total_events': 0,
            'events_with_cells': 0,
            'events_after_min_cells_filter': 0,
            'total_cells_before_filtering': 0,
            'total_cells_after_filtering': 0
        }
        
        with h5py.File(file_path, 'r') as f:
            vertex_data = f['HSvertex'][:]
            cells_data = f['cells'][:]
            
            file_stats['total_events'] = len(vertex_data)
            
            for i in range(len(vertex_data)):
                # Extract vertex features for global context
                if self.config.use_spatial_features:
                    vertex_reco = [
                        vertex_data[i]['HSvertex_reco_x'],
                        vertex_data[i]['HSvertex_reco_y'],
                        vertex_data[i]['HSvertex_reco_z']
                    ]
                else:
                    vertex_reco = [0.0, 0.0, 0.0]
                
                # Process cells for this event
                event_cells = cells_data[i]
                
                if len(event_cells) > 0:
                    file_stats['events_with_cells'] += 1
                    file_stats['total_cells_before_filtering'] += len(event_cells)
                
                # Apply configurable cell filtering
                valid_cells = self.apply_cell_filtering(event_cells)
                
                if len(valid_cells) > 0:
                    file_stats['total_cells_after_filtering'] += len(valid_cells)
                
                # Skip events with too few cells
                if len(valid_cells) < self.config.min_cells:
                    continue
                
                file_stats['events_after_min_cells_filter'] += 1
                
                # Process cells for this event
                sequence = self._process_event_cells(valid_cells)
                if sequence is None:
                    continue
                
                cell_sequences.append(sequence)
                vertex_features.append(vertex_reco)
                vertex_times.append(vertex_data[i]['HSvertex_time'])
                sequence_lengths.append(len(sequence))
        
        return cell_sequences, vertex_features, vertex_times, sequence_lengths, file_stats
    
    def _process_event_cells(self, valid_cells: np.ndarray) -> Optional[List[List[float]]]:
        """Process cells for a single event."""
        # Sort cells by selection feature
        if self.config.cell_selection_feature in ['Cell_e', 'Cell_significance', 'matched_track_pt']:
            sorted_indices = np.argsort(-valid_cells[self.config.cell_selection_feature])
        else:
            sorted_indices = np.argsort(-valid_cells[self.config.cell_selection_feature])
        
        # Take up to max_cells
        n_cells_to_use = min(len(valid_cells), self.config.max_cells)
        sorted_cells = valid_cells[sorted_indices[:n_cells_to_use]]
        
        # Create sequence of cell features
        sequence = []
        for cell_idx in range(n_cells_to_use):
            cell_features_values = []
            for feature in self.config.cell_features:
                if feature in sorted_cells.dtype.names:
                    cell_features_values.append(sorted_cells[feature][cell_idx])
                else:
                    print(f"Warning: Feature '{feature}' not found in cell data. Using 0.0 as default.")
                    cell_features_values.append(0.0)
            sequence.append(cell_features_values)
        
        return sequence
    
    def _print_filtering_statistics(self, stats: dict):
        """Print detailed cell filtering statistics."""
        print("\n" + "="*60)
        print("CELL FILTERING STATISTICS")
        print("="*60)
        
        print(f"Events:")
        print(f"  Total events processed: {stats['total_events']}")
        print(f"  Events with cells: {stats['events_with_cells']}")
        print(f"  Events after min_cells filter: {stats['events_after_min_cells_filter']}")
        
        if stats['events_with_cells'] > 0:
            event_retention_rate = (stats['events_after_min_cells_filter'] / stats['events_with_cells']) * 100
            print(f"  Event retention rate: {event_retention_rate:.1f}%")
        
        print(f"\nCells:")
        print(f"  Total cells before filtering: {stats['total_cells_before_filtering']}")
        print(f"  Total cells after filtering: {stats['total_cells_after_filtering']}")
        
        if stats['total_cells_before_filtering'] > 0:
            cell_retention_rate = (stats['total_cells_after_filtering'] / stats['total_cells_before_filtering']) * 100
            cells_removed = stats['total_cells_before_filtering'] - stats['total_cells_after_filtering']
            print(f"  Cells removed: {cells_removed}")
            print(f"  Cell retention rate: {cell_retention_rate:.1f}%")
            
            if stats['events_after_min_cells_filter'] > 0:
                avg_cells_per_event = stats['total_cells_after_filtering'] / stats['events_after_min_cells_filter']
                print(f"  Average cells per event (after filtering): {avg_cells_per_event:.1f}")
        
        print(f"\nFiltering Configuration:")
        print(f"  {self.config.get_cell_filtering_description()}")
        print("="*60)
    
    def _print_sequence_statistics(self, sequence_lengths: np.ndarray):
        """Print statistics about sequence lengths."""
        if len(sequence_lengths) == 0:
            print("No valid sequences found!")
            return
            
        print(f"\nSequence length statistics:")
        print(f"  Mean: {np.mean(sequence_lengths):.2f}")
        print(f"  Std: {np.std(sequence_lengths):.2f}")
        print(f"  Min: {np.min(sequence_lengths)}")
        print(f"  Max: {np.max(sequence_lengths)}")
        print(f"  Median: {np.median(sequence_lengths):.2f}")
        
        # Show distribution of sequence lengths
        unique_lengths, counts = np.unique(sequence_lengths, return_counts=True)
        print(f"  Most common lengths:")
        for length, count in sorted(zip(unique_lengths, counts), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(sequence_lengths)) * 100
            print(f"    Length {length}: {count} events ({percentage:.1f}%)")