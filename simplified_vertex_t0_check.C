#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TCanvas.h>

const double c_light = 0.299792458; // mm/ps

// Calibration parameters from original code
const float emb1_y[7] = {48.5266, 37.56, 28.9393, 23.1505, 18.5468, 13.0141, 8.03724};
const float emb1_ysigma[7] = {416.994, 293.206, 208.321, 148.768, 117.756, 106.804, 57.6545};

const float emb2_y[7] = {46.2244, 41.5079, 38.5544, 36.9812, 31.2718, 29.7469, 19.331};
const float emb2_ysigma[7] = {2001.56, 1423.38, 1010.24, 720.392, 551.854, 357.594, 144.162};

const float emb3_y[7] = {104.325, 106.119, 71.1017, 75.151, 51.2334, 48.2088, 46.6502};
const float emb3_ysigma[7] = {1215.53, 880.826, 680.742, 468.689, 372.184, 279.134, 162.288};

const float eme1_y[7] = {125.348, 102.888, 86.7558, 59.7355, 55.3299, 41.3032, 23.646};
const float eme1_ysigma[7] = {855.662, 589.529, 435.052, 314.788, 252.453, 185.536, 76.5333};

const float eme2_y[7] = {272.149, 224.475, 173.443, 135.829, 113.05, 83.8009, 37.1829};
const float eme2_ysigma[7] = {1708.6, 1243.34, 881.465, 627.823, 486.99, 311.032, 106.533};

const float eme3_y[7] = {189.356, 140.293, 111.232, 86.8784, 69.0834, 60.5034, 38.5008};
const float eme3_ysigma[7] = {1137.06, 803.044, 602.152, 403.393, 318.327, 210.827, 99.697};

// Histograms for plotting
TH1F *truthTimeHist;
TH1F *recoTimeHist;
TH1F *errorTimeHist;
TH2F *recoVsTruthHist;

// Global counters for debugging
int eventCounter = 0;
int validEvents = 0;

void initialize_histograms() {
    const int bins = 400;
    const double min_range = -2000;
    const double max_range = 2000;

    truthTimeHist = new TH1F("truthTime", "Truth Vertex Time", bins, min_range, max_range);
    truthTimeHist->GetXaxis()->SetTitle("Truth Time [ps]");
    truthTimeHist->GetYaxis()->SetTitle("Events");
    
    recoTimeHist = new TH1F("recoTime", "Reconstructed t0", bins, min_range, max_range);
    recoTimeHist->GetXaxis()->SetTitle("Reconstructed Time [ps]");
    recoTimeHist->GetYaxis()->SetTitle("Events");
    
    errorTimeHist = new TH1F("errorTime", "t0 Error (Reco - Truth)", bins, min_range, max_range);
    errorTimeHist->GetXaxis()->SetTitle("Error [ps]");
    errorTimeHist->GetYaxis()->SetTitle("Events");

    recoVsTruthHist = new TH2F("recoVsTruth", "Reconstructed vs Truth t0", 
                               bins, min_range, max_range, bins, min_range, max_range);
    recoVsTruthHist->GetXaxis()->SetTitle("Truth Time [ps]");
    recoVsTruthHist->GetYaxis()->SetTitle("Reconstructed Time [ps]");
}

float get_mean(bool is_barrel, int layer, int energy_bin) {
    if (is_barrel) {
        if (layer == 1) return emb1_y[energy_bin];
        else if (layer == 2) return emb2_y[energy_bin];
        else if (layer == 3) return emb3_y[energy_bin];
    } else { // is_endcap
        if (layer == 1) return eme1_y[energy_bin];
        else if (layer == 2) return eme2_y[energy_bin];
        else if (layer == 3) return eme3_y[energy_bin];
    }
    return 0.0;
}

float get_sigma(bool is_barrel, int layer, int energy_bin) {
    if (is_barrel) {
        if (layer == 1) return emb1_ysigma[energy_bin];
        else if (layer == 2) return emb2_ysigma[energy_bin];
        else if (layer == 3) return emb3_ysigma[energy_bin];
    } else { // is_endcap
        if (layer == 1) return eme1_ysigma[energy_bin];
        else if (layer == 2) return eme2_ysigma[energy_bin];
        else if (layer == 3) return eme3_ysigma[energy_bin];
    }
    return 1.0;
}

int get_energy_bin(float energy) {
    if (energy > 1 && energy <= 1.5) return 0;
    else if (energy > 1.5 && energy <= 2) return 1;
    else if (energy > 2 && energy <= 3) return 2;
    else if (energy > 3 && energy <= 4) return 3;
    else if (energy > 4 && energy <= 5) return 4;
    else if (energy > 5 && energy <= 10) return 5;
    else if (energy > 10) return 6;
    return -1; // Invalid bin
}

void process_file(const std::string &filename, float energyThreshold = 1.0, float significancecut = 4.0) {
    TFile *file = TFile::Open(filename.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    TTree *tree = nullptr;
    file->GetObject("ntuple", tree);
    if (!tree) {
        std::cerr << "Error getting TTree 'ntuple' from file: " << filename << std::endl;
        file->Close();
        return;
    }

    // Branch variables (simplified - only what we need)
    std::vector<float> *truthVtxTime = nullptr;
    std::vector<float> *truthVtxX = nullptr;
    std::vector<float> *truthVtxY = nullptr;
    std::vector<float> *truthVtxZ = nullptr;
    std::vector<float> *recoVtxX = nullptr;
    std::vector<float> *recoVtxY = nullptr;
    std::vector<float> *recoVtxZ = nullptr;
    std::vector<bool> *recoVtxIsHS = nullptr;
    std::vector<bool> *truthVtxIsHS = nullptr;
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

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        tree->GetEntry(entry);

        for (size_t i = 0; i < truthVtxTime->size(); ++i) {
            if (!truthVtxIsHS->at(i)) continue;
            
            eventCounter++;
            float vtx_time = truthVtxTime->at(i);
            float vtx_x = truthVtxX->at(i);
            float vtx_y = truthVtxY->at(i);
            float vtx_z = truthVtxZ->at(i);
            
            truthTimeHist->Fill(vtx_time);

            // Find matching reco vertex
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

            double weighted_sum = 0.0, weight_sum = 0.0;
            std::vector<float> adjusted_cell_times; // For debugging output
            int cells_used = 0;

            for (size_t j = 0; j < cellE->size(); ++j) {
                if (cellE->at(j) < energyThreshold) continue;
                if (cellSignificance->at(j) < significancecut) continue;

                float cell_time = cellTime->at(j);
                float cell_x = cellX->at(j);
                float cell_y = cellY->at(j);
                float cell_z = cellZ->at(j);
                float cell_eta = cellEta->at(j);
                float cell_phi = cellPhi->at(j);

                // Distance corrections
                float distance_to_origin = std::sqrt(cell_x*cell_x + cell_y*cell_y + cell_z*cell_z);
                float distance_vtx_to_cell = std::sqrt((cell_x - reco_vtx_x)*(cell_x - reco_vtx_x)
                                                     + (cell_y - reco_vtx_y)*(cell_y - reco_vtx_y)
                                                     + (cell_z - reco_vtx_z)*(cell_z - reco_vtx_z));
                float corrected_time = cell_time + distance_to_origin / c_light - distance_vtx_to_cell / c_light;

                bool is_barrel = cellIsEMBarrel->at(j);
                bool is_endcap = cellIsEMEndCap->at(j);
                int layer = cellLayer->at(j);
                float energy = cellE->at(j);

                int bin = get_energy_bin(energy);
                if (bin == -1 || (!is_barrel && !is_endcap) || layer < 1 || layer > 3) continue;

                // Track matching (simplified version)
                bool matched_track_HS = false;
                float matched_track_pt = -999;
                
                for (size_t k = 0; k < trackPt->size(); ++k) {
                    if (trackQuality->at(k) == 0) continue;
                    
                    float DeltaR = 999;
                    std::vector<float>* trackExtrapolatedEta = nullptr;
                    std::vector<float>* trackExtrapolatedPhi = nullptr;

                    // Select appropriate track extrapolation based on detector and layer
                    if (is_barrel) {
                        if (layer == 1) {
                            trackExtrapolatedEta = trackExtrapolatedEta_EMB1;
                            trackExtrapolatedPhi = trackExtrapolatedPhi_EMB1;
                        } else if (layer == 2) {
                            trackExtrapolatedEta = trackExtrapolatedEta_EMB2;
                            trackExtrapolatedPhi = trackExtrapolatedPhi_EMB2;
                        } else if (layer == 3) {
                            trackExtrapolatedEta = trackExtrapolatedEta_EMB3;
                            trackExtrapolatedPhi = trackExtrapolatedPhi_EMB3;
                        }
                    } else if (is_endcap) {
                        if (layer == 1) {
                            trackExtrapolatedEta = trackExtrapolatedEta_EME1;
                            trackExtrapolatedPhi = trackExtrapolatedPhi_EME1;
                        } else if (layer == 2) {
                            trackExtrapolatedEta = trackExtrapolatedEta_EME2;
                            trackExtrapolatedPhi = trackExtrapolatedPhi_EME2;
                        } else if (layer == 3) {
                            trackExtrapolatedEta = trackExtrapolatedEta_EME3;
                            trackExtrapolatedPhi = trackExtrapolatedPhi_EME3;
                        }
                    }

                    if (trackExtrapolatedEta != nullptr && trackExtrapolatedPhi != nullptr) {
                        float dEta = trackExtrapolatedEta->at(k) - cell_eta;
                        float dPhi = trackExtrapolatedPhi->at(k) - cell_phi;
                        if (dPhi >= M_PI) dPhi -= 2 * M_PI;
                        else if (dPhi < -M_PI) dPhi += 2 * M_PI;
                        DeltaR = std::sqrt(dEta * dEta + dPhi * dPhi);
                    }

                    if (DeltaR > 0.05) continue;
                    
                    if (trackPt->at(k) > matched_track_pt) {
                        matched_track_pt = trackPt->at(k);
                        matched_track_HS = (TrackftagTruthOrigin->at(k) != 0);
                    }
                }

                if (!matched_track_HS) continue;

                // Apply calibration
                float mean = get_mean(is_barrel, layer, bin);
                float sigma = get_sigma(is_barrel, layer, bin);
                
                float adjusted_time = corrected_time - mean;
                float weight = 1.0 / (sigma * sigma);
                
                weighted_sum += adjusted_time * weight;
                weight_sum += weight;
                adjusted_cell_times.push_back(adjusted_time);
                cells_used++;
            }

            if (weight_sum > 0) {
                float event_reco_time = weighted_sum / weight_sum;
                float error = event_reco_time - vtx_time;
                
                recoTimeHist->Fill(event_reco_time);
                errorTimeHist->Fill(error);
                recoVsTruthHist->Fill(vtx_time, event_reco_time);
                validEvents++;
                
                // Print debug info for first 10 events
                if (validEvents <= 10) {
                    std::cout << "\nEvent " << validEvents << ":" << std::endl;
                    std::cout << "  Truth vertex time: " << std::fixed << std::setprecision(4) << vtx_time << " ps" << std::endl;
                    std::cout << "  Number of filtered cells: " << cells_used << std::endl;
                    std::cout << "  Adjusted cell times: ";
                    for (size_t idx = 0; idx < std::min(adjusted_cell_times.size(), size_t(5)); ++idx) {
                        std::cout << std::fixed << std::setprecision(4) << adjusted_cell_times[idx];
                        if (idx < std::min(adjusted_cell_times.size(), size_t(5)) - 1) std::cout << ", ";
                    }
                    if (adjusted_cell_times.size() > 5) std::cout << "...";
                    std::cout << std::endl;
                    std::cout << "  Reconstructed vertex time: " << std::fixed << std::setprecision(4) << event_reco_time << " ps" << std::endl;
                    std::cout << "  Error (reco - truth): " << std::fixed << std::setprecision(4) << error << " ps" << std::endl;
                }
            }
        }
    }

    file->Close();
    delete file;
    std::cout << "Processed file: " << filename << std::endl;
}

void save_plots() {
    // Create output directory
    std::filesystem::create_directories("cpp_method_check_output");
    
    // Create canvases and save plots
    TCanvas *c1 = new TCanvas("c1", "Truth Time Distribution", 800, 600);
    truthTimeHist->Draw();
    c1->SaveAs("cpp_method_check_output/true_vertex_time_distribution.png");
    delete c1;
    
    TCanvas *c2 = new TCanvas("c2", "Reconstructed Time Distribution", 800, 600);
    recoTimeHist->Draw();
    c2->SaveAs("cpp_method_check_output/reconstructed_t0_distribution.png");
    delete c2;
    
    TCanvas *c3 = new TCanvas("c3", "Error Distribution", 800, 600);
    errorTimeHist->Draw();
    c3->SaveAs("cpp_method_check_output/t0_error_distribution.png");
    delete c3;
    
    TCanvas *c4 = new TCanvas("c4", "2D Histogram", 800, 600);
    recoVsTruthHist->Draw("COLZ");
    c4->SaveAs("cpp_method_check_output/reconstructed_t0_vs_true_2d.png");
    delete c4;
    
    std::cout << "Plots saved to cpp_method_check_output/" << std::endl;
}

void simplified_vertex_t0_check(float energyThreshold = 1.0, float significancecut = 4.0, int startIndex = 1, int endIndex = 5) {
    eventCounter = 0;
    validEvents = 0;
    
    initialize_histograms();
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "SIMPLIFIED C++ VERTEX T0 RECONSTRUCTION CHECK" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Energy threshold: " << energyThreshold << " GeV" << std::endl;
    std::cout << "Significance cut: " << significancecut << std::endl;
    std::cout << "Processing files " << startIndex << " to " << endIndex << std::endl;

    const std::string path = "/fs/ddn/sdf/group/atlas/d/sanha/data_storage/upgrade/user.scheong.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.SuperNtuple.e8514_s4345_r15583.20250511_Output";
    for (int i = startIndex; i <= endIndex; ++i) {
        std::ostringstream filename;
        filename << path << "/user.scheong.44627907.Output._" 
                 << std::setw(6) << std::setfill('0') << i 
                 << ".SuperNtuple.root";

        if (std::filesystem::exists(filename.str())) {
            process_file(filename.str(), energyThreshold, significancecut);
        } else {
            std::cerr << "File does not exist: " << filename.str() << std::endl;
        }
    }

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "PROCESSING SUMMARY" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Total truth vertices processed: " << eventCounter << std::endl;
    std::cout << "Valid events (with reconstruction): " << validEvents << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
              << (100.0 * validEvents / eventCounter) << "%" << std::endl;
    
    // Print basic statistics
    if (validEvents > 0) {
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Truth time - Mean: " << std::fixed << std::setprecision(2) 
                  << truthTimeHist->GetMean() << " ps, RMS: " << truthTimeHist->GetRMS() << " ps" << std::endl;
        std::cout << "  Reco time - Mean: " << std::fixed << std::setprecision(2) 
                  << recoTimeHist->GetMean() << " ps, RMS: " << recoTimeHist->GetRMS() << " ps" << std::endl;
        std::cout << "  Error - Mean: " << std::fixed << std::setprecision(2) 
                  << errorTimeHist->GetMean() << " ps, RMS: " << errorTimeHist->GetRMS() << " ps" << std::endl;
    }
    
    // Save plots
    save_plots();
    
    // Cleanup
    delete truthTimeHist;
    delete recoTimeHist;
    delete errorTimeHist;
    delete recoVsTruthHist;
    
    std::cout << "\nC++ method check completed!" << std::endl;
}

// Main function for standalone execution
int main(int argc, char* argv[]) {
    float energyThreshold = 1.0;
    float significancecut = 4.0;
    int startIndex = 1;
    int endIndex = 1;
    
    if (argc > 1) energyThreshold = std::atof(argv[1]);
    if (argc > 2) significancecut = std::atof(argv[2]);
    if (argc > 3) startIndex = std::atoi(argv[3]);
    if (argc > 4) endIndex = std::atoi(argv[4]);
    
    simplified_vertex_t0_check(energyThreshold, significancecut, startIndex, endIndex);
    
    return 0;
}
