"""Data loader for HGTD-only models using only HGTD tracks (no LAr data)."""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
from config.base_config import BaseConfig
from .data_loader import DataLoader


class HGTDOnlyDataLoader(DataLoader):
    """Load and preprocess data for HGTD-only models (no LAr cells, jets, or tracks)."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _process_event_hgtd_tracks(self, event_hgtd_tracks: np.ndarray) -> List[List[float]]:
        """Process HGTD tracks for a single event."""
        # Filter: valid == True and Track_hasValidTime == 1
        valid_mask = (event_hgtd_tracks['valid'] == True) & (event_hgtd_tracks['Track_hasValidTime'] == 1)
        valid_hgtd_tracks = event_hgtd_tracks[valid_mask]

        # Sort by pt in descending order
        sorted_indices = np.argsort(-valid_hgtd_tracks['Track_pt'])
        sorted_hgtd_tracks = valid_hgtd_tracks[sorted_indices]

        # Select top N tracks (max_hgtd_tracks from config)
        n_hgtd_tracks_to_use = min(len(sorted_hgtd_tracks), self.config.max_hgtd_tracks)

        # Map config HGTD track features to HDF5 field names
        feature_mapping = {
            'pt': 'Track_pt',
            'eta': 'Track_eta',
            'phi': 'Track_phi',
            'd0': 'Track_d0',
            'z0': 'Track_z0',
            'time': 'Track_time',
            'timeRes': 'Track_timeRes'
        }

        hgtd_track_sequence = []
        for track_idx in range(n_hgtd_tracks_to_use):
            track = sorted_hgtd_tracks[track_idx]
            hgtd_track_features = []
            for feature in self.config.hgtd_track_features:
                hdf5_field = feature_mapping[feature]
                hgtd_track_features.append(track[hdf5_field])
            hgtd_track_sequence.append(hgtd_track_features)

        return hgtd_track_sequence

    def load_data_from_files(self, file_paths: Optional[List[str]] = None) -> Tuple:
        """
        Load HGTD-only data (no LAr cells, jets, or tracks).

        Returns:
            Tuple of (hgtd_track_sequences, vertex_features, vertex_times, sequence_lengths)
        """
        if file_paths is None:
            file_paths = self.get_file_paths()

        all_hgtd_track_sequences = []
        all_vertex_features = []
        all_vertex_times = []
        sequence_lengths = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue

            with h5py.File(file_path, 'r') as f:
                vertex_data = f['HSvertex'][:]
                hgtd_tracks_data = f['tracks_HGTD'][:]

                for i in range(len(vertex_data)):
                    # Process HGTD tracks
                    hgtd_track_sequence = self._process_event_hgtd_tracks(hgtd_tracks_data[i])

                    # Check if we have minimum required HGTD tracks
                    if len(hgtd_track_sequence) < self.config.min_hgtd_tracks:
                        continue

                    # Vertex features based on spatial features configuration
                    if self.config.use_spatial_features:
                        vertex_reco = [
                            vertex_data[i]['HSvertex_reco_x'],
                            vertex_data[i]['HSvertex_reco_y'],
                            vertex_data[i]['HSvertex_reco_z']
                        ]
                    else:
                        vertex_reco = [0.0, 0.0, 0.0]
                    vertex_time = vertex_data[i]['HSvertex_time']

                    all_hgtd_track_sequences.append(hgtd_track_sequence)
                    all_vertex_features.append(vertex_reco)
                    all_vertex_times.append(vertex_time)
                    sequence_lengths.append(len(hgtd_track_sequence))

        return (all_hgtd_track_sequences, np.array(all_vertex_features),
                np.array(all_vertex_times), sequence_lengths)
