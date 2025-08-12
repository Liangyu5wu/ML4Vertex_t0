#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <TFile.h>
#include <TTree.h>

const double c_light = 0.299792458; // mm/ps

// Cell-track matching calibration parameters (original)
const float emb1_y_track[7] = {48.5266, 37.56, 28.9393, 23.1505, 18.5468, 13.0141, 8.03724};
const float emb1_ysigma_track[7] = {416.994, 293.206, 208.321, 148.768, 117.756, 106.804, 57.6545};
const float emb2_y_track[7] = {46.2244, 41.5079, 38.5544, 36.9812, 31.2718, 29.7469, 19.331};
const float emb2_ysigma_track[7] = {2001.56, 1423.38, 1010.24, 720.392, 551.854, 357.594, 144.162};
const float emb3_y_track[7] = {104.325, 106.119, 71.1017, 75.151, 51.2334, 48.2088, 46.6502};
const float emb3_ysigma_track[7] = {1215.53, 880.826, 680.742, 468.689, 372.184, 279.134, 162.288};
const float eme1_y_track[7] = {125.348, 102.888, 86.7558, 59.7355, 55.3299, 41.3032, 23.646};
const float eme1_ysigma_track[7] = {855.662, 589.529, 435.052, 314.788, 252.453, 185.536, 76.5333};
const float eme2_y_track[7] = {272.149, 224.475, 173.443, 135.829, 113.05, 83.8009, 37.1829};
const float eme2_ysigma_track[7] = {1708.6, 1243.34, 881.465, 627.823, 486.99, 311.032, 106.533};
const float eme3_y_track[7] = {189.356, 140.293, 111.232, 86.8784, 69.0834, 60.5034, 38.5008};
const float eme3_ysigma_track[7] = {1137.06, 803.044, 602.152, 403.393, 318.327, 210.827, 99.697};

// Cell-jet matching calibration parameters (from reference code)
const float emb1_y_jet[7] = {44.0004, 36.1672, 27.7946, 22.8566, 18.6052, 13.5979, 8.63018};
const float emb1_ysigma_jet[7] = {428.908, 299.433, 210.949, 149.052, 117.804, 107.251, 57.5941};
const float emb2_y_jet[7] = {-37.6927, -4.76502, 18.4064, 28.5538, 23.8885, 26.6632, 18.3774};
const float emb2_ysigma_jet[7] = {2167.48, 1507, 1056.7, 736.56, 562.19, 360.442, 145.808};
const float emb3_y_jet[7] = {76.9224, 96.6823, 61.8036, 59.5018, 60.5852, 42.9381, 42.8869};
const float emb3_ysigma_jet[7] = {1328.7, 937.662, 707.828, 488.065, 405.723, 280.358, 166.422};
const float eme1_y_jet[7] = {110.994, 91.5183, 77.769, 60.8697, 52.896, 39.3309, 20.7378};
const float eme1_ysigma_jet[7] = {1076.77, 768.73, 570.185, 404.887, 301.699, 207.33, 104.583};
const float eme2_y_jet[7] = {140.013, 135.814, 118.308, 101.373, 87.0022, 71.9089, 37.0023};
const float eme2_ysigma_jet[7] = {1759.24, 1295.54, 939.963, 680.803, 537.385, 353.075, 112.837};
const float eme3_y_jet[7] = {125.077, 103.004, 88.0972, 68.0425, 54.7863, 56.1797, 38.8708};
const float eme3_ysigma_jet[7] = {1197.23, 856.656, 625.128, 439.311, 356.805, 224.769, 101.193};

float get_mean(bool is_barrel, int layer, int energy_bin, bool use_jet_matching) {
    if (use_jet_matching) {
        if (is_barrel) {
            if (layer == 1) return emb1_y_jet[energy_bin];
            else if (layer == 2) return emb2_y_jet[energy_bin];
            else if (layer == 3) return emb3_y_jet[energy_bin];
        } else {
            if (layer == 1) return eme1_y_jet[energy_bin];
            else if (layer == 2) return eme2_y_jet[energy_bin];
            else if (layer == 3) return eme3_y_jet[energy_bin];
        }
    } else {
        if (is_barrel) {
            if (layer == 1) return emb1_y_track[energy_bin];
            else if (layer == 2) return emb2_y_track[energy_bin];
            else if (layer == 3) return emb3_y_track[energy_bin];
        } else {
            if (layer == 1) return eme1_y_track[energy_bin];
            else if (layer == 2) return eme2_y_track[energy_bin];
            else if (layer == 3) return eme3_y_track[energy_bin];
        }
    }
    return 0.0;
}

float get_sigma(bool is_barrel, int layer, int energy_bin, bool use_jet_matching) {
    if (use_jet_matching) {
        if (is_barrel) {
            if (layer == 1) return emb1_ysigma_jet[energy_bin];
            else if (layer == 2) return emb2_ysigma_jet[energy_bin];
            else if (layer == 3) return emb3_ysigma_jet[energy_bin];
        } else {
            if (layer == 1) return eme1_ysigma_jet[energy_bin];
            else if (layer == 2) return eme2_ysigma_jet[energy_bin];
            else if (layer == 3) return eme3_ysigma_jet[energy_bin];
        }
    } else {
        if (is_barrel) {
            if (layer == 1) return emb1_ysigma_track[energy_bin];
            else if (layer == 2) return emb2_ysigma_track[energy_bin];
            else if (layer == 3) return emb3_ysigma_track[energy_bin];
        } else {
            if (layer == 1) return eme1_ysigma_track[energy_bin];
            else if (layer == 2) return eme2_ysigma_track[energy_bin];
            else if (layer == 3) return eme3_ysigma_track[energy_bin];
        }
    }
    return 1.0;
}

int process_file(const std::string &filename, float energyThreshold = 1.0, float significancecut = 4.0, 
                bool use_jet_matching = false, float deltaRThreshold = 0.3, 
                float jetPtMin = 30.0, float jetPtMax = 1000.0) {
    
    // Generate ROOT dictionaries for jet matching
    if (use_jet_matching) {
        gInterpreter->GenerateDictionary("vector<vector<int> >", "vector");
    }
    
    TFile *file = TFile::Open(filename.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 0;
    }

    TTree *tree = nullptr;
    file->GetObject("ntuple", tree);
    if (!tree) {
        std::cerr << "Error getting TTree 'ntuple' from file: " << filename << std::endl;
        file->Close();
        return 0;
    }

    // Branch variables
    std::vector<float> *truthVtxTime = nullptr;
    std::vector<float> *truthVtxX = nullptr;
    std::vector<float> *truthVtxY = nullptr;
    std::vector<float> *truthVtxZ = nullptr;
    std::vector<float> *recoVtxX = nullptr;
    std::vector<float> *recoVtxY = nullptr;
    std::vector<float> *recoVtxZ = nullptr;
    std::vector<bool> *recoVtxIsHS = nullptr;
    std::vector<bool> *truthVtxIsHS = nullptr;
    ULong64_t eventNumber = 0;
    std::vector<float> *cellTime = nullptr;
    std::vector<float> *cellE = nullptr;
    std::vector<float> *cellX = nullptr;
    std::vector<float> *cellY = nullptr;
    std::vector<float> *cellZ = nullptr;
    std::vector<float> *cellEta = nullptr;
    std::vector<float> *cellPhi = nullptr;
    std::vector<bool> *cellIsEMBarrel = nullptr;
    std::vector<bool> *cellIsEMEndCap = nullptr;
    std::vector<int> *cellLayer = nullptr;
    std::vector<float> *cellSignificance = nullptr;
    std::vector<float> *trackPt = nullptr;
    std::vector<int> *trackQuality = nullptr;
    std::vector<float> *trackExtrapolatedEta_EMB1 = nullptr;
    std::vector<float> *trackExtrapolatedPhi_EMB1 = nullptr;
    std::vector<float> *trackExtrapolatedEta_EMB2 = nullptr;
    std::vector<float> *trackExtrapolatedPhi_EMB2 = nullptr;
    std::vector<float> *trackExtrapolatedEta_EMB3 = nullptr;
    std::vector<float> *trackExtrapolatedPhi_EMB3 = nullptr;
    std::vector<float> *trackExtrapolatedEta_EME1 = nullptr;
    std::vector<float> *trackExtrapolatedPhi_EME1 = nullptr;
    std::vector<float> *trackExtrapolatedEta_EME2 = nullptr;
    std::vector<float> *trackExtrapolatedPhi_EME2 = nullptr;
    std::vector<float> *trackExtrapolatedEta_EME3 = nullptr;
    std::vector<float> *trackExtrapolatedPhi_EME3 = nullptr;
    std::vector<float> *TrackftagTruthOrigin = nullptr;
    
    // Jet variables (for jet matching)
    std::vector<float> *TopoJetsPt = nullptr;
    std::vector<float> *TopoJetsEta = nullptr;
    std::vector<float> *TopoJetsPhi = nullptr;
    std::vector<std::vector<int>> *TopoJets_TruthHSJetIdx = nullptr;

    // Set branch addresses
    tree->SetBranchAddress("TruthVtx_time", &truthVtxTime);
    tree->SetBranchAddress("TruthVtx_x", &truthVtxX);
    tree->SetBranchAddress("TruthVtx_y", &truthVtxY);
    tree->SetBranchAddress("TruthVtx_z", &truthVtxZ);
    tree->SetBranchAddress("TruthVtx_isHS", &truthVtxIsHS);
    tree->SetBranchAddress("RecoVtx_x", &recoVtxX);
    tree->SetBranchAddress("RecoVtx_y", &recoVtxY);
    tree->SetBranchAddress("RecoVtx_z", &recoVtxZ);
    tree->SetBranchAddress("RecoVtx_isHS", &recoVtxIsHS);
    tree->SetBranchAddress("Cell_time", &cellTime);
    tree->SetBranchAddress("Cell_e", &cellE);
    tree->SetBranchAddress("Cell_x", &cellX);
    tree->SetBranchAddress("Cell_y", &cellY);
    tree->SetBranchAddress("Cell_z", &cellZ);
    tree->SetBranchAddress("Cell_eta", &cellEta);
    tree->SetBranchAddress("Cell_phi", &cellPhi);
    tree->SetBranchAddress("Cell_isEM_Barrel", &cellIsEMBarrel);
    tree->SetBranchAddress("Cell_isEM_EndCap", &cellIsEMEndCap);
    tree->SetBranchAddress("Cell_layer", &cellLayer);
    tree->SetBranchAddress("Cell_significance", &cellSignificance);
    
    if (use_jet_matching) {
        tree->SetBranchAddress("AntiKt4EMTopoJets_pt", &TopoJetsPt);
        tree->SetBranchAddress("AntiKt4EMTopoJets_eta", &TopoJetsEta);
        tree->SetBranchAddress("AntiKt4EMTopoJets_phi", &TopoJetsPhi);
        tree->SetBranchAddress("AntiKt4EMTopoJets_truthHSJet_idx", &TopoJets_TruthHSJetIdx);
    } else {
        tree->SetBranchAddress("Track_pt", &trackPt);
        tree->SetBranchAddress("Track_quality", &trackQuality);
        tree->SetBranchAddress("Track_EMB1_eta", &trackExtrapolatedEta_EMB1);
        tree->SetBranchAddress("Track_EMB1_phi", &trackExtrapolatedPhi_EMB1);
        tree->SetBranchAddress("Track_EMB2_eta", &trackExtrapolatedEta_EMB2);
        tree->SetBranchAddress("Track_EMB2_phi", &trackExtrapolatedPhi_EMB2);
        tree->SetBranchAddress("Track_EMB3_eta", &trackExtrapolatedEta_EMB3);
        tree->SetBranchAddress("Track_EMB3_phi", &trackExtrapolatedPhi_EMB3);
        tree->SetBranchAddress("Track_EME1_eta", &trackExtrapolatedEta_EME1);
        tree->SetBranchAddress("Track_EME1_phi", &trackExtrapolatedPhi_EME1);
        tree->SetBranchAddress("Track_EME2_eta", &trackExtrapolatedEta_EME2);
        tree->SetBranchAddress("Track_EME2_phi", &trackExtrapolatedPhi_EME2);
        tree->SetBranchAddress("Track_EME3_eta", &trackExtrapolatedEta_EME3);
        tree->SetBranchAddress("Track_EME3_phi", &trackExtrapolatedPhi_EME3);
        tree->SetBranchAddress("Track_ftagTruthOrigin", &TrackftagTruthOrigin);
    }

    Long64_t nEntries = tree->GetEntries();
    int processed_events = 0;
    
    std::cout << "\n=== VERTEX TIME RECONSTRUCTION DEBUG OUTPUT (" 
              << (use_jet_matching ? "JET" : "TRACK") << " MATCHING) ===" << std::endl;
    
    for (Long64_t entry = 0; entry < nEntries && processed_events < 10; ++entry) {
        tree->GetEntry(entry);

        for (size_t i = 0; i < truthVtxTime->size(); ++i) {
            if (!truthVtxIsHS->at(i)) continue;
            
            float truth_vtx_time = truthVtxTime->at(i);
            
            // Find reconstructed vertex
            bool foundRecoVtx = false;
            float reco_vtx_x = 0.0, reco_vtx_y = 0.0, reco_vtx_z = 0.0;
            for (size_t reco_i = 0; reco_i < recoVtxIsHS->size(); ++reco_i) {
                if (!recoVtxIsHS->at(reco_i)) continue;
                reco_vtx_x = recoVtxX->at(reco_i);
                reco_vtx_y = recoVtxY->at(reco_i);
                reco_vtx_z = recoVtxZ->at(reco_i);
                foundRecoVtx = true;
                break;
            }
            
            if (!foundRecoVtx) continue;

            std::cout << "\n--- Event " << (processed_events + 1) << " ---" << std::endl;
            std::cout << "Truth vertex time: " << std::fixed << std::setprecision(2) << truth_vtx_time << " ps" << std::endl;

            double weighted_sum = 0.0, weight_sum = 0.0;
            std::vector<float> calibrated_cell_times;
            std::vector<float> original_cell_times;
            int filtered_cells = 0;

            // Get selected jets for jet matching
            std::vector<size_t> selectedJetIndices;
            if (use_jet_matching && TopoJetsPt && TopoJets_TruthHSJetIdx) {
                for (size_t j = 0; j < TopoJetsPt->size(); ++j) {
                    bool isInPtRange = (TopoJetsPt->at(j) >= jetPtMin && TopoJetsPt->at(j) <= jetPtMax);
                    bool hasMatch = (j < TopoJets_TruthHSJetIdx->size() && !TopoJets_TruthHSJetIdx->at(j).empty());
                    if (isInPtRange && hasMatch) {
                        selectedJetIndices.push_back(j);
                    }
                }
            }

            // Process cells
            for (size_t j = 0; j < cellE->size(); ++j) {
                if (cellE->at(j) < energyThreshold) continue;
                if (cellSignificance->at(j) < significancecut) continue;

                float cell_time = cellTime->at(j);
                float cell_x = cellX->at(j);
                float cell_y = cellY->at(j);
                float cell_z = cellZ->at(j);
                float cell_eta = cellEta->at(j);
                float cell_phi = cellPhi->at(j);

                // Time-of-flight correction
                float distance_to_origin = std::sqrt(cell_x*cell_x + cell_y*cell_y + cell_z*cell_z);
                float distance_vtx_to_cell = std::sqrt((cell_x - reco_vtx_x)*(cell_x - reco_vtx_x)
                                                     + (cell_y - reco_vtx_y)*(cell_y - reco_vtx_y)
                                                     + (cell_z - reco_vtx_z)*(cell_z - reco_vtx_z));
                float corrected_time = cell_time + distance_to_origin / c_light - distance_vtx_to_cell / c_light;

                bool is_barrel = cellIsEMBarrel->at(j);
                bool is_endcap = cellIsEMEndCap->at(j);
                int layer = cellLayer->at(j);
                float energy = cellE->at(j);

                // Energy binning
                int bin = -1;
                if (energy > 1 && energy <= 1.5) bin = 0;
                else if (energy > 1.5 && energy <= 2) bin = 1;
                else if (energy > 2 && energy <= 3) bin = 2;
                else if (energy > 3 && energy <= 4) bin = 3;
                else if (energy > 4 && energy <= 5) bin = 4;
                else if (energy > 5 && energy <= 10) bin = 5;
                else if (energy > 10) bin = 6;

                if (bin == -1 || (!is_barrel && !is_endcap) || layer < 1 || layer > 3) continue;

                bool has_match = false;

                if (use_jet_matching) {
                    // Check jet matching
                    for (size_t jetIdx : selectedJetIndices) {
                        if (jetIdx < TopoJetsEta->size() && jetIdx < TopoJetsPhi->size()) {
                            float jetEta = TopoJetsEta->at(jetIdx);
                            float jetPhi = TopoJetsPhi->at(jetIdx);
                            
                            float dEta = jetEta - cell_eta;
                            float dPhi = jetPhi - cell_phi;
                            if (dPhi >= M_PI) dPhi -= 2 * M_PI;
                            else if (dPhi < -M_PI) dPhi += 2 * M_PI;
                            float deltaR = std::sqrt(dEta * dEta + dPhi * dPhi);
                            
                            if (deltaR <= deltaRThreshold) {
                                has_match = true;
                                break;
                            }
                        }
                    }
                } else {
                    // Track matching (original logic)
                    for (size_t k = 0; k < trackPt->size(); ++k) {
                        if (trackQuality->at(k) == 0) continue;

                        std::vector<float>* trackEta = nullptr;
                        std::vector<float>* trackPhi = nullptr;

                        if (is_barrel) {
                            if (layer == 1) { trackEta = trackExtrapolatedEta_EMB1; trackPhi = trackExtrapolatedPhi_EMB1; }
                            else if (layer == 2) { trackEta = trackExtrapolatedEta_EMB2; trackPhi = trackExtrapolatedPhi_EMB2; }
                            else if (layer == 3) { trackEta = trackExtrapolatedEta_EMB3; trackPhi = trackExtrapolatedPhi_EMB3; }
                        } else if (is_endcap) {
                            if (layer == 1) { trackEta = trackExtrapolatedEta_EME1; trackPhi = trackExtrapolatedPhi_EME1; }
                            else if (layer == 2) { trackEta = trackExtrapolatedEta_EME2; trackPhi = trackExtrapolatedPhi_EME2; }
                            else if (layer == 3) { trackEta = trackExtrapolatedEta_EME3; trackPhi = trackExtrapolatedPhi_EME3; }
                        }

                        if (!trackEta || !trackPhi) continue;

                        float dEta = trackEta->at(k) - cell_eta;
                        float dPhi = trackPhi->at(k) - cell_phi;
                        if (dPhi >= M_PI) dPhi -= 2 * M_PI;
                        else if (dPhi < -M_PI) dPhi += 2 * M_PI;
                        float deltaR = std::sqrt(dEta * dEta + dPhi * dPhi);

                        if (deltaR <= 0.05 && TrackftagTruthOrigin->at(k) != 0) {
                            has_match = true;
                            break;
                        }
                    }
                }

                if (!has_match) continue;

                // Apply calibration
                float mean = get_mean(is_barrel, layer, bin, use_jet_matching);
                float sigma = get_sigma(is_barrel, layer, bin, use_jet_matching);
                float adjusted_time = corrected_time - mean;
                float weight = 1.0 / (sigma * sigma);

                weighted_sum += adjusted_time * weight;
                weight_sum += weight;
                
                // Store both original (TOF corrected) and calibrated times
                original_cell_times.push_back(corrected_time);
                calibrated_cell_times.push_back(adjusted_time);
                filtered_cells++;
            }

            std::cout << "Number of filtered cells: " << filtered_cells << std::endl;
            
            if (filtered_cells > 0) {
                // Print all original cell times (before calibration)
                std::cout << "Original cell times (before calibration): ";
                for (size_t k = 0; k < original_cell_times.size(); ++k) {
                    std::cout << std::fixed << std::setprecision(1) << original_cell_times[k];
                    if (k < original_cell_times.size() - 1) std::cout << ", ";
                }
                std::cout << " ps" << std::endl;
                
                // Print all calibrated cell times (after calibration)
                std::cout << "Calibrated cell times (after calibration): ";
                for (size_t k = 0; k < calibrated_cell_times.size(); ++k) {
                    std::cout << std::fixed << std::setprecision(1) << calibrated_cell_times[k];
                    if (k < calibrated_cell_times.size() - 1) std::cout << ", ";
                }
                std::cout << " ps" << std::endl;
                
                if (weight_sum > 0) {
                    float reco_vtx_time = weighted_sum / weight_sum;
                    float error = reco_vtx_time - truth_vtx_time;
                    
                    std::cout << "Reconstructed vertex time: " << std::fixed << std::setprecision(2) << reco_vtx_time << " ps" << std::endl;
                    std::cout << "Error (reco - truth): " << std::fixed << std::setprecision(2) << error << " ps" << std::endl;
                } else {
                    std::cout << "No valid reconstruction (weight_sum = 0)" << std::endl;
                }
            } else {
                std::cout << "No cells passed filtering criteria" << std::endl;
            }

            processed_events++;
            if (processed_events >= 10) break;
        }
    }

    file->Close();
    delete file;
    
    std::cout << "\nProcessed " << processed_events << " events from: " << filename << std::endl;
    
    return processed_events;
}

void processmu200_reco_simplified(float energyThreshold = 1.0, float significancecut = 4.0, int fileIndex = 21, 
                                 bool use_jet_matching = false, float deltaRThreshold = 0.3, 
                                 float jetPtMin = 30.0, float jetPtMax = 100000.0) {
    const std::string path = "../";
    std::ostringstream filename;
    filename << path << "/user.scheong.43348828.Output._" 
             << std::setw(6) << std::setfill('0') << fileIndex 
             << ".SuperNtuple.root";

    if (std::filesystem::exists(filename.str())) {
        std::cout << "Processing file: " << filename.str() << std::endl;
        std::cout << "Energy threshold: " << energyThreshold << " GeV" << std::endl;
        std::cout << "Significance cut: " << significancecut << std::endl;
        std::cout << "Matching type: " << (use_jet_matching ? "Jet" : "Track") << std::endl;
        
        if (use_jet_matching) {
            std::cout << "Jet pT range: " << jetPtMin << " - " << jetPtMax << " GeV" << std::endl;
            std::cout << "Delta R threshold: " << deltaRThreshold << std::endl;
        }
        
        int processed_events = process_file(filename.str(), energyThreshold, significancecut, 
                                          use_jet_matching, deltaRThreshold, jetPtMin, jetPtMax);
        
        std::cout << "\nCompleted processing " << processed_events << " events." << std::endl;
        
    } else {
        std::cerr << "File does not exist: " << filename.str() << std::endl;
    }
}
