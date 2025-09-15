#include "SwiftF0.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

// Simple WAV file reader structure
struct WavHeader {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};

SwiftF0::SwiftF0(float confidence_threshold, float fmin, float fmax,
                 const std::string& model_path)
    : confidence_threshold_(confidence_threshold), fmin_(fmin), fmax_(fmax) {
    
    // Validate parameters
    if (confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
        throw std::invalid_argument("confidence_threshold must be between 0.0 and 1.0");
    }
    
    if (fmin < MODEL_FMIN) {
        throw std::invalid_argument("fmin is below model minimum");
    }
    
    if (fmax > MODEL_FMAX) {
        throw std::invalid_argument("fmax is above model maximum");
    }
    
    if (fmin > fmax) {
        throw std::invalid_argument("fmin cannot be greater than fmax");
    }
    
    // Initialize ONNX Runtime
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SwiftF0");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetInterOpNumThreads(1);
    
    // Create session
    #ifdef _WIN32
        std::wstring wide_model_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wide_model_path.c_str(), *session_options_);
    #else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
    #endif
    
    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t num_input_nodes = session_->GetInputCount();
    input_names_ptr_.reserve(num_input_nodes);
    input_names_.reserve(num_input_nodes);
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
        input_names_ptr_.push_back(std::move(input_name));
    }
    
    size_t num_output_nodes = session_->GetOutputCount();
    output_names_ptr_.reserve(num_output_nodes);
    output_names_.reserve(num_output_nodes);
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
        output_names_ptr_.push_back(std::move(output_name));
    }
    
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

SwiftF0::~SwiftF0() = default;

std::vector<float> SwiftF0::resample(const std::vector<float>& audio,
                                     int orig_sr, int target_sr) {
    if (orig_sr == target_sr) {
        return audio;
    }
    
    // Simple linear interpolation resampling
    double ratio = static_cast<double>(target_sr) / orig_sr;
    size_t new_length = static_cast<size_t>(audio.size() * ratio);
    std::vector<float> resampled(new_length);
    
    for (size_t i = 0; i < new_length; i++) {
        double src_idx = i / ratio;
        size_t idx_low = static_cast<size_t>(std::floor(src_idx));
        size_t idx_high = std::min(idx_low + 1, audio.size() - 1);
        double frac = src_idx - idx_low;
        
        resampled[i] = static_cast<float>(
            audio[idx_low] * (1.0 - frac) + audio[idx_high] * frac);
    }
    
    return resampled;
}

std::pair<std::vector<float>, std::vector<float>> 
SwiftF0::extract_pitch_and_confidence(const std::vector<float>& audio_16k) {
    
    std::vector<float> padded_audio = audio_16k;
    
    // Pad audio if needed
    if (padded_audio.size() < MIN_AUDIO_LENGTH) {
        padded_audio.resize(MIN_AUDIO_LENGTH, 0.0f);
    }
    
    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(padded_audio.size())};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, padded_audio.data(), padded_audio.size(),
        input_shape.data(), input_shape.size());
    
    // Run inference
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_names_.data(), &input_tensor, 1,
                                        output_names_.data(), output_names_.size());
    
    if (output_tensors.size() < 2) {
        throw std::runtime_error("Model returned insufficient outputs");
    }
    
    // Extract pitch and confidence
    float* pitch_data = output_tensors[0].GetTensorMutableData<float>();
    float* confidence_data = output_tensors[1].GetTensorMutableData<float>();
    
    auto pitch_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto conf_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t n_frames = pitch_shape.back();
    
    std::vector<float> pitch_hz(pitch_data, pitch_data + n_frames);
    std::vector<float> confidence(confidence_data, confidence_data + n_frames);
    
    return {pitch_hz, confidence};
}

std::vector<bool> SwiftF0::compute_voicing(const std::vector<float>& pitch_hz,
                                           const std::vector<float>& confidence) {
    std::vector<bool> voicing(pitch_hz.size());
    
    for (size_t i = 0; i < pitch_hz.size(); i++) {
        voicing[i] = (confidence[i] > confidence_threshold_) &&
                     (pitch_hz[i] >= fmin_) &&
                     (pitch_hz[i] <= fmax_);
    }
    
    return voicing;
}

std::vector<float> SwiftF0::calculate_timestamps(size_t n_frames) {
    std::vector<float> timestamps(n_frames);
    
    for (size_t i = 0; i < n_frames; i++) {
        float frame_center = i * HOP_LENGTH + CENTER_OFFSET;
        timestamps[i] = frame_center / TARGET_SAMPLE_RATE;
    }
    
    return timestamps;
}

PitchResult SwiftF0::detect_from_array(const std::vector<float>& audio_array,
                                       int sample_rate) {
    if (audio_array.empty()) {
        throw std::invalid_argument("Input audio cannot be empty");
    }
    
    if (sample_rate <= 0) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    
    // Resample to 16kHz if needed
    std::vector<float> audio_16k = resample(audio_array, sample_rate, TARGET_SAMPLE_RATE);
    
    // Extract pitch and confidence
    auto [pitch_hz, confidence] = extract_pitch_and_confidence(audio_16k);
    
    // Compute voicing decisions
    std::vector<bool> voicing = compute_voicing(pitch_hz, confidence);
    
    // Generate timestamps
    std::vector<float> timestamps = calculate_timestamps(pitch_hz.size());
    
    return PitchResult{pitch_hz, confidence, timestamps, voicing};
}

PitchResult SwiftF0::detect_from_file(const std::string& audio_path) {
    // Read WAV file
    std::ifstream file(audio_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open audio file: " + audio_path);
    }
    
    // Read WAV header
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    
    // Basic validation
    if (std::string(header.riff, 4) != "RIFF" || 
        std::string(header.wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV file format");
    }
    
    // Read audio data
    size_t num_samples = header.data_size / (header.bits_per_sample / 8);
    std::vector<float> audio_data(num_samples);
    
    if (header.bits_per_sample == 16) {
        std::vector<int16_t> raw_data(num_samples);
        file.read(reinterpret_cast<char*>(raw_data.data()), header.data_size);
        
        // Convert to float and normalize
        for (size_t i = 0; i < num_samples; i++) {
            audio_data[i] = raw_data[i] / 32768.0f;
        }
    } else if (header.bits_per_sample == 32) {
        file.read(reinterpret_cast<char*>(audio_data.data()), header.data_size);
    } else {
        throw std::runtime_error("Unsupported bit depth: " + 
                                std::to_string(header.bits_per_sample));
    }
    
    // Convert to mono if needed
    if (header.num_channels > 1) {
        std::vector<float> mono_audio(num_samples / header.num_channels);
        for (size_t i = 0; i < mono_audio.size(); i++) {
            float sum = 0.0f;
            for (uint16_t ch = 0; ch < header.num_channels; ch++) {
                sum += audio_data[i * header.num_channels + ch];
            }
            mono_audio[i] = sum / header.num_channels;
        }
        audio_data = mono_audio;
    }
    
    file.close();
    
    return detect_from_array(audio_data, header.sample_rate);
}