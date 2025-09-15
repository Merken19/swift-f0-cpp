#include "SwiftF0.h"
#include <iostream>
#include <iomanip>
#include <exception>

void print_results(const PitchResult& result) {
    std::cout << "SwiftF0 Pitch Detection Results\n";
    std::cout << "================================\n";
    std::cout << "Total frames: " << result.timestamps.size() << "\n\n";
    
    // Count voiced frames
    size_t voiced_count = 0;
    for (bool v : result.voicing) {
        if (v) voiced_count++;
    }
    std::cout << "Voiced frames: " << voiced_count << " / " << result.timestamps.size() 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * voiced_count / result.timestamps.size()) << "%)\n\n";
    
    // Calculate statistics for voiced frames
    if (voiced_count > 0) {
        float min_pitch = 10000.0f, max_pitch = 0.0f, avg_pitch = 0.0f;
        float avg_confidence = 0.0f;
        
        for (size_t i = 0; i < result.pitch_hz.size(); i++) {
            if (result.voicing[i]) {
                min_pitch = std::min(min_pitch, result.pitch_hz[i]);
                max_pitch = std::max(max_pitch, result.pitch_hz[i]);
                avg_pitch += result.pitch_hz[i];
                avg_confidence += result.confidence[i];
            }
        }
        
        avg_pitch /= voiced_count;
        avg_confidence /= voiced_count;
        
        std::cout << "Voiced frames statistics:\n";
        std::cout << "  Min pitch: " << std::fixed << std::setprecision(2) 
                  << min_pitch << " Hz\n";
        std::cout << "  Max pitch: " << max_pitch << " Hz\n";
        std::cout << "  Avg pitch: " << avg_pitch << " Hz\n";
        std::cout << "  Avg confidence: " << std::fixed << std::setprecision(4) 
                  << avg_confidence << "\n\n";
    }
    
    // Print first 1000 frames
    std::cout << "First 1000 frames:\n";
    std::cout << std::setw(12) << "Time (s)" 
              << std::setw(12) << "Pitch (Hz)"
              << std::setw(12) << "Confidence"
              << std::setw(10) << "Voiced\n";
    std::cout << std::string(46, '-') << "\n";
    
    size_t max_frames = std::min(size_t(1000), result.timestamps.size());
    for (size_t i = 0; i < max_frames; i++) {
        std::cout << std::fixed << std::setprecision(4) 
                  << std::setw(12) << result.timestamps[i]
                  << std::setprecision(2) 
                  << std::setw(12) << result.pitch_hz[i]
                  << std::setprecision(4)
                  << std::setw(12) << result.confidence[i]
                  << std::setw(10) << (result.voicing[i] ? "true" : "false")
                  << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        // Default parameters
        std::string audio_file = "recorded_samples.wav";
        float fmin = 46.875f;
        float fmax = 2093.75f;
        float confidence_threshold = 0.9f;
        
        // Parse command line arguments
        if (argc > 1) {
            audio_file = argv[1];
        }
        if (argc > 2) {
            fmin = std::stof(argv[2]);
        }
        if (argc > 3) {
            fmax = std::stof(argv[3]);
        }
        if (argc > 4) {
            confidence_threshold = std::stof(argv[4]);
        }
        
        std::cout << "SwiftF0 C++ Implementation\n";
        std::cout << "==========================\n";
        std::cout << "Audio file: " << audio_file << "\n";
        std::cout << "Frequency range: " << fmin << " - " << fmax << " Hz\n";
        std::cout << "Confidence threshold: " << confidence_threshold << "\n\n";
        
        // Initialize the detector
        SwiftF0 detector(confidence_threshold, fmin, fmax, "model.onnx");
        
        // Run pitch detection
        std::cout << "Processing audio...\n\n";
        PitchResult result = detector.detect_from_file(audio_file);
        
        // Print results
        print_results(result);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}