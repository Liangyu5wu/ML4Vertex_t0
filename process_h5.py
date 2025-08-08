import os
import h5py
import numpy as np
import math
import argparse
from pathlib import Path
from tqdm import tqdm

# python process_h5.py --input-dir ./Vertex_timing --output-dir ./selected_h5 --end-idx 0 --max-events 20

def compute_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def compute_delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = phi1 - phi2
    if dphi >= np.pi:
        dphi -= 2 * np.pi
    elif dphi < -np.pi:
        dphi += 2 * np.pi
    return np.sqrt(deta**2 + dphi**2)

def match_track_to_cell(cell_eta, cell_phi, is_barrel, cell_layer, track_data, track_valid_mask):
    matched_track_DeltaR = 999.0
    matched_track_pt = -999.0
    matched_track_HS = False
    deltaRThreshold = 0.05

    valid_track_indices = np.where(track_valid_mask)[0]
    
    for k in valid_track_indices:
        DeltaR = 999.0
        track_eta = None
        track_phi = None
        
        if is_barrel:
            if cell_layer == 3:  # EMB3
                track_eta = track_data[k]['Track_EMB3_eta']
                track_phi = track_data[k]['Track_EMB3_phi']
            elif cell_layer == 2:  # EMB2
                track_eta = track_data[k]['Track_EMB2_eta']
                track_phi = track_data[k]['Track_EMB2_phi']
            elif cell_layer == 1:  # EMB1
                track_eta = track_data[k]['Track_EMB1_eta']
                track_phi = track_data[k]['Track_EMB1_phi']
        else:  # EndCap
            if cell_layer == 3:  # EME3
                track_eta = track_data[k]['Track_EME3_eta']
                track_phi = track_data[k]['Track_EME3_phi']
            elif cell_layer == 2:  # EME2
                track_eta = track_data[k]['Track_EME2_eta']
                track_phi = track_data[k]['Track_EME2_phi']
            elif cell_layer == 1:  # EME1
                track_eta = track_data[k]['Track_EME1_eta']
                track_phi = track_data[k]['Track_EME1_phi']
        
        if track_eta is not None and track_phi is not None:
            dEta = track_eta - cell_eta
            dPhi = track_phi - cell_phi
            if dPhi >= math.pi:
                dPhi -= 2 * math.pi
            elif dPhi < -math.pi:
                dPhi += 2 * math.pi
            DeltaR = math.sqrt(dEta * dEta + dPhi * dPhi)
        
        if DeltaR > deltaRThreshold:
            continue
            
        if track_data[k]['Track_pt'] > matched_track_pt:
            matched_track_DeltaR = DeltaR
            matched_track_pt = track_data[k]['Track_pt']
            matched_track_HS = (track_data[k]['Track_isGoodFromHS'] == 1)
    
    is_matched_hs = 1 if (matched_track_pt > 0 and matched_track_HS) else 0
    matched_pt = matched_track_pt if matched_track_pt > 0 else 0
    matched_deltaR = matched_track_DeltaR if matched_track_pt > 0 else 999.0
    
    return is_matched_hs, matched_pt, matched_deltaR

def process_h5_file(input_file, output_file, max_events=None):

    print(f"Processing {input_file} -> {output_file}")
    
    with h5py.File(input_file, 'r') as f_in:
        hs_vertex = f_in['HSvertex'][:]
        
        if max_events is not None and max_events > 0 and max_events < len(hs_vertex):
            print(f"Limiting to first {max_events} events of {len(hs_vertex)} total events")
            hs_vertex = hs_vertex[:max_events]
        
        distance = compute_distance(
            hs_vertex['HSvertex_x'], hs_vertex['HSvertex_y'], hs_vertex['HSvertex_z'],
            hs_vertex['HSvertex_reco_x'], hs_vertex['HSvertex_reco_y'], hs_vertex['HSvertex_reco_z']
        )
        valid_events_mask = distance <= 2.0
        valid_event_indices = np.where(valid_events_mask)[0]
        
        print(f"Found {len(valid_event_indices)} valid events out of {len(hs_vertex)}")
        
        if len(valid_event_indices) == 0:
            print("No valid events found. Skipping file.")
            return
        
        cells_data = f_in['cells'][valid_event_indices]
        tracks_data = f_in['tracks'][valid_event_indices]
        
        with h5py.File(output_file, 'w') as f_out:

            hs_vertex_dtype = np.dtype([
                ('HSvertex_time', np.float32),
                ('HSvertex_reco_x', np.float32),
                ('HSvertex_reco_y', np.float32),
                ('HSvertex_reco_z', np.float32),
                ('eventNumber', np.int32)
            ])
            hs_vertex_out = np.empty(len(valid_event_indices), dtype=hs_vertex_dtype)
            
            for i, idx in enumerate(valid_event_indices):
                hs_vertex_out[i]['HSvertex_time'] = hs_vertex[idx]['HSvertex_time']
                hs_vertex_out[i]['HSvertex_reco_x'] = hs_vertex[idx]['HSvertex_reco_x']
                hs_vertex_out[i]['HSvertex_reco_y'] = hs_vertex[idx]['HSvertex_reco_y']
                hs_vertex_out[i]['HSvertex_reco_z'] = hs_vertex[idx]['HSvertex_reco_z']
                hs_vertex_out[i]['eventNumber'] = hs_vertex[idx]['eventNumber']
            
            f_out.create_dataset('HSvertex', data=hs_vertex_out)
            
            cells_dtype = np.dtype([
                ('Cell_time', np.float64),
                ('valid', np.bool_),
                ('Cell_time_TOF_corrected', np.float64),
                ('Cell_e', np.float64),
                ('Cell_x', np.float64),
                ('Cell_y', np.float64),
                ('Cell_z', np.float64),
                ('Cell_eta', np.float64),
                ('Cell_phi', np.float64),
                ('Cell_Barrel', np.int32),   
                ('Cell_layer', np.int32),  
                ('Cell_significance', np.float64),
                ('Sig_above_3_celle_above_1GeV', np.int32),
                ('matched_track_HS', np.int32),
                ('matched_track_pt', np.float64),
                ('matched_track_deltaR', np.float64)
            ])
            
            processed_cells = np.zeros((len(valid_event_indices), 1000), dtype=cells_dtype)

            valid_cell_counts = np.zeros(len(valid_event_indices), dtype=np.int32)
            matched_hs_cell_counts = np.zeros(len(valid_event_indices), dtype=np.int32)
            

            for event_idx, global_event_idx in enumerate(tqdm(valid_event_indices, desc="Processing events")):
                event_cells = cells_data[event_idx]
                event_tracks = tracks_data[event_idx]
                
                valid_cells_mask = (event_cells['valid'] == True) & (event_cells['Sig_above_3_celle_above_1GeV'] == 1)
                valid_cells = event_cells[valid_cells_mask]
                
                if len(valid_cells) == 0:
                    continue
                
                valid_cell_counts[event_idx] = len(valid_cells)
                valid_tracks_mask = (event_tracks['valid'] == True) & (event_tracks['Track_isGoodFromHS'] == 1)

                matched_hs_count = 0
                
                for i, cell in enumerate(valid_cells):
                    if i >= 1000:
                        break
                        
                    processed_cells[event_idx, i]['Cell_time'] = cell['Cell_time']
                    processed_cells[event_idx, i]['valid'] = cell['valid']
                    processed_cells[event_idx, i]['Cell_time_TOF_corrected'] = cell['Cell_time_TOF_corrected']
                    processed_cells[event_idx, i]['Cell_e'] = cell['Cell_e']
                    processed_cells[event_idx, i]['Cell_x'] = cell['Cell_x']
                    processed_cells[event_idx, i]['Cell_y'] = cell['Cell_y']
                    processed_cells[event_idx, i]['Cell_z'] = cell['Cell_z']
                    processed_cells[event_idx, i]['Cell_eta'] = cell['Cell_eta']
                    processed_cells[event_idx, i]['Cell_phi'] = cell['Cell_phi']
                    processed_cells[event_idx, i]['Cell_significance'] = cell['Cell_significance']
                    processed_cells[event_idx, i]['Sig_above_3_celle_above_1GeV'] = cell['Sig_above_3_celle_above_1GeV']
                    
                    is_barrel = (cell['Cell_isEM_Barrel'] == 1)
                    processed_cells[event_idx, i]['Cell_Barrel'] = 1 if is_barrel else 0
                    processed_cells[event_idx, i]['Cell_layer'] = cell['Cell_layer'] + 1
                    
                    matched_hs, matched_pt, matched_deltaR = match_track_to_cell(
                        cell['Cell_eta'], cell['Cell_phi'], 
                        is_barrel,
                        processed_cells[event_idx, i]['Cell_layer'],
                        event_tracks, valid_tracks_mask
                    )
                    
                    processed_cells[event_idx, i]['matched_track_HS'] = matched_hs
                    processed_cells[event_idx, i]['matched_track_pt'] = matched_pt
                    processed_cells[event_idx, i]['matched_track_deltaR'] = matched_deltaR

                    if matched_hs == 1:
                        matched_hs_count += 1
                matched_hs_cell_counts[event_idx] = matched_hs_count
            
            f_out.create_dataset(
                'cells', 
                data=processed_cells,
                compression="gzip", 
                compression_opts=9
            )

            events_with_cells_mask = valid_cell_counts > 0
            events_with_cells_count = np.sum(events_with_cells_mask)
            
            print(f"Saved cells data with shape {processed_cells.shape}")
            print(f"Max cells per event: {np.max(valid_cell_counts)}")
            print(f"Min cells per event (of events with cells): {np.min(valid_cell_counts[events_with_cells_mask]) if events_with_cells_count > 0 else 0}")

            events_with_matched_hs_cells_mask = matched_hs_cell_counts > 0
            events_with_matched_hs_cells_count = np.sum(events_with_matched_hs_cells_mask)
            
            if events_with_matched_hs_cells_count > 0:
                max_matched_hs_cells = np.max(matched_hs_cell_counts)
                min_matched_hs_cells = np.min(matched_hs_cell_counts[events_with_matched_hs_cells_mask])
                avg_matched_hs_cells = np.mean(matched_hs_cell_counts[events_with_matched_hs_cells_mask])
                
                print(f"Events with HS-matched cells: {events_with_matched_hs_cells_count} out of {len(valid_event_indices)} ({events_with_matched_hs_cells_count/len(valid_event_indices)*100:.2f}%)")
                print(f"Max HS-matched cells per event: {max_matched_hs_cells}")
                print(f"Min HS-matched cells per event (of events with HS-matched cells): {min_matched_hs_cells}")
                print(f"Avg HS-matched cells per event (of events with HS-matched cells): {avg_matched_hs_cells:.2f}")

                all_matched_pts = []
                all_matched_deltaRs = []
                for event_idx in range(len(valid_event_indices)):
                    valid_cell_count = valid_cell_counts[event_idx]
                    if valid_cell_count > 0:
                        for i in range(valid_cell_count):
                            if processed_cells[event_idx, i]['matched_track_HS'] == 1:
                                all_matched_pts.append(processed_cells[event_idx, i]['matched_track_pt'])
                                all_matched_deltaRs.append(processed_cells[event_idx, i]['matched_track_deltaR'])
                
                if all_matched_pts:
                    all_matched_pts = np.array(all_matched_pts)
                    all_matched_deltaRs = np.array(all_matched_deltaRs)
                    
                    print(f"Matched track pt statistics:")
                    print(f"  Min: {np.min(all_matched_pts):.2f} GeV")
                    print(f"  Max: {np.max(all_matched_pts):.2f} GeV")
                    print(f"  Mean: {np.mean(all_matched_pts):.2f} GeV")
                    print(f"  Median: {np.median(all_matched_pts):.2f} GeV")
                    
                    print(f"Matched track deltaR statistics:")
                    print(f"  Min: {np.min(all_matched_deltaRs):.5f}")
                    print(f"  Max: {np.max(all_matched_deltaRs):.5f}")
                    print(f"  Mean: {np.mean(all_matched_deltaRs):.5f}")
                    print(f"  Median: {np.median(all_matched_deltaRs):.5f}")
            else:
                print("No events with HS-matched cells found.")
            
            f_out.create_dataset(
                'tracks', 
                data=tracks_data,
                compression="gzip", 
                compression_opts=9
            )
            
            print(f"Saved {len(tracks_data)} tracks for valid events")

def main():

    parser = argparse.ArgumentParser(description='Process H5 files for ML training.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input H5 files')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save processed H5 files')
    parser.add_argument('--file-pattern', type=str, default='output_*.h5', help='Pattern to match H5 files (default: output_*.h5)')
    parser.add_argument('--start-idx', type=int, default=0, help='Starting file index (default: 0)')
    parser.add_argument('--end-idx', type=int, default=49, help='Ending file index (inclusive, default: 49)')
    parser.add_argument('--max-events', type=int, default=None, help='Maximum number of events to process per file (default: all)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    input_dir = Path(args.input_dir)
    
    for i in range(args.start_idx, args.end_idx + 1):
        input_file = input_dir / f"output_{i:03d}.h5"
        output_file = output_dir / f"output_{i:03d}.h5"
        
        if input_file.exists():
            try:
                process_h5_file(input_file, output_file, args.max_events)
                print(f"Completed: {output_file}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
        else:
            print(f"File {input_file} not found. Skipping.")

if __name__ == "__main__":
    main()
