"""Data loader for HGTD multi-input models with LAr cells, jets, tracks, and HGTD tracks."""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
from config.base_config import BaseConfig
from .multi_input_data_loader import MultiInputDataLoader


class HGTDMultiInputDataLoader(MultiInputDataLoader):
    """Load and preprocess data for HGTD multi-input models with LAr cells, jets, tracks, and HGTD tracks."""

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

        # Pad to max_hgtd_tracks
        while len(hgtd_track_sequence) < self.config.max_hgtd_tracks:
            padding_values = []
            for feature in self.config.hgtd_track_features:
                padding_values.append(self.config.hgtd_track_padding_values[feature])
            hgtd_track_sequence.append(padding_values)

        return hgtd_track_sequence

    def load_data_from_files(self, file_paths: Optional[List[str]] = None) -> Tuple:
        """Load HGTD multi-input data with LAr cells, jets, tracks, and HGTD tracks."""
        if file_paths is None:
            file_paths = self.get_file_paths()

        all_cell_sequences = []
        all_vertex_features = []
        all_vertex_times = []
        all_jet_sequences = []
        all_track_sequences = []
        all_hgtd_track_sequences = []
        sequence_lengths = []

        # Diagnostic counters
        total_events = 0
        events_after_cell_filtering = 0
        events_after_cell_processing = 0
        events_loaded = 0

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue

            with h5py.File(file_path, 'r') as f:
                vertex_data = f['HSvertex'][:]
                cells_data = f['cells'][:]
                jets_data = f['jets'][:]
                tracks_data = f['tracks'][:]
                hgtd_tracks_data = f['tracks_HGTD'][:]

                for i in range(len(vertex_data)):
                    total_events += 1

                    try:
                        # Process cells
                        event_cells = cells_data[i]
                        valid_cells = self.apply_cell_filtering(event_cells)

                        if len(valid_cells) < self.config.min_cells:
                            continue

                        events_after_cell_filtering += 1

                        # Process cells sequence
                        cell_sequence = self._process_event_cells(valid_cells)
                        if cell_sequence is None:
                            continue

                        events_after_cell_processing += 1

                        # Process jets
                        jet_sequence = self._process_event_jets(jets_data[i])

                        # Process LAr tracks
                        track_sequence = self._process_event_tracks(tracks_data[i])

                        # Process HGTD tracks (NEW)
                        hgtd_track_sequence = self._process_event_hgtd_tracks(hgtd_tracks_data[i])
                    except Exception as e:
                        print(f"Error processing event {i} in {os.path.basename(file_path)}: {e}")
                        import traceback
                        traceback.print_exc()
                        raise  # Re-raise to see the full error

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

                    all_cell_sequences.append(cell_sequence)
                    all_vertex_features.append(vertex_reco)
                    all_vertex_times.append(vertex_time)
                    all_jet_sequences.append(jet_sequence)
                    all_track_sequences.append(track_sequence)
                    all_hgtd_track_sequences.append(hgtd_track_sequence)
                    sequence_lengths.append(len(cell_sequence))
                    events_loaded += 1

        # Print diagnostic information
        print(f"\n{'='*70}")
        print(f"HGTD Multi-Input Data Loading Statistics:")
        print(f"{'='*70}")
        print(f"Total events processed:           {total_events}")
        print(f"Events after cell filtering:      {events_after_cell_filtering} ({100*events_after_cell_filtering/total_events if total_events > 0 else 0:.1f}%)")
        print(f"Events after cell processing:     {events_after_cell_processing} ({100*events_after_cell_processing/total_events if total_events > 0 else 0:.1f}%)")
        print(f"Events successfully loaded:       {events_loaded} ({100*events_loaded/total_events if total_events > 0 else 0:.1f}%)")
        print(f"{'='*70}\n")

        if events_loaded == 0:
            print("⚠️  WARNING: 0 events loaded!")
            if events_after_cell_filtering == 0:
                print("   → All events filtered out by cell filtering (time quality cut or min_cells)")
                print(f"   → Config: use_time_quality_cut={self.config.use_time_quality_cut}, min_cells={self.config.min_cells}")
            elif events_after_cell_processing == 0:
                print("   → All events filtered out during cell processing")

        return (all_cell_sequences, np.array(all_vertex_features),
                np.array(all_vertex_times), sequence_lengths,
                all_jet_sequences, all_track_sequences, all_hgtd_track_sequences)
