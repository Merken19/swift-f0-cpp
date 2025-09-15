#ifndef SWIFT_F0_H
#define SWIFT_F0_H

#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

struct PitchResult {
    std::vector<float> pitch_hz;
    std::vector<float> confidence;
    std::vector<float> timestamps;
    std::vector<bool> voicing;
};

class SwiftF0 {
public:
    // Audio processing constants
    static constexpr int TARGET_SAMPLE_RATE = 16000;
    static constexpr int HOP_LENGTH = 256;
    static constexpr int FRAME_LENGTH = 1024;
    static constexpr int STFT_PADDING = (FRAME_LENGTH - HOP_LENGTH) / 2;  // 384
    static constexpr int MIN_AUDIO_LENGTH = 256;
    static constexpr float CENTER_OFFSET = (FRAME_LENGTH - 1) / 2.0f - STFT_PADDING;  // 127.5
    
    // Model frequency limits
    static constexpr float MODEL_FMIN = 46.875f;
    static constexpr float MODEL_FMAX = 2093.75f;
    
    // Default parameters
    static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.9f;

    SwiftF0(float confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD,
            float fmin = MODEL_FMIN,
            float fmax = MODEL_FMAX,
            const std::string& model_path = "model.onnx");
    
    ~SwiftF0();
    
    PitchResult detect_from_file(const std::string& audio_path);
    PitchResult detect_from_array(const std::vector<float>& audio_array, int sample_rate);

private:
    float confidence_threshold_;
    float fmin_;
    float fmax_;
    
    // ONNX Runtime session
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // Model input/output names
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr_;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr_;
    
    std::pair<std::vector<float>, std::vector<float>> extract_pitch_and_confidence(
        const std::vector<float>& audio_16k);
    
    std::vector<bool> compute_voicing(const std::vector<float>& pitch_hz,
                                      const std::vector<float>& confidence);
    
    std::vector<float> calculate_timestamps(size_t n_frames);
    
    std::vector<float> resample(const std::vector<float>& audio,
                                int orig_sr, int target_sr);
};

#endif // SWIFT_F0_H