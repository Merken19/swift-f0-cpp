# SwiftF0 C++ Implementation

A C++ port of the SwiftF0 pitch detection library using ONNX Runtime.

## Project Structure

```
swift_f0_cpp/
├── SwiftF0.h              # Header file with class definitions
├── SwiftF0.cpp            # Implementation of SwiftF0 class
├── main.cpp               # Main program
├── CMakeLists.txt         # CMake build configuration
├── build.bat              # Windows build script
├── model.onnx             # ONNX model file (you need to copy this)
├── recorded_samples.wav   # Your input audio file
└── nuget_packages/        # ONNX Runtime libraries
    └── Microsoft.ML.OnnxRuntime.1.22.1/
        ├── build/
        │   └── native/
        │       └── include/   # Header files
        └── runtimes/
            └── win-x64/
                └── native/    # DLL and LIB files
```

## Prerequisites

1. **Visual Studio 2022** (or 2019) with C++ development tools
2. **CMake** (version 3.13 or higher)
3. **ONNX Runtime** NuGet package extracted to `nuget_packages/`

## Setup Instructions

### 1. Install ONNX Runtime

Download and extract the ONNX Runtime NuGet package:

```powershell
# Using PowerShell
cd swift_f0_cpp
mkdir nuget_packages
cd nuget_packages

# Download ONNX Runtime (version 1.20.1)
Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.20.1" -OutFile "onnxruntime.zip"

# Extract the package
Expand-Archive -Path "onnxruntime.zip" -DestinationPath "Microsoft.ML.OnnxRuntime.1.20.1"
Remove-Item "onnxruntime.zip"
cd ..
```

Alternatively, you can use the NuGet CLI:
```cmd
nuget install Microsoft.ML.OnnxRuntime -Version 1.20.1 -OutputDirectory nuget_packages
```

### 2. Copy Required Files

1. Copy `model.onnx` from your Python swift_f0 package to the `swift_f0_cpp/` directory
2. Place your `recorded_samples.wav` file in the `swift_f0_cpp/` directory

### 3. Build the Project

#### Option A: Using the build script (Recommended)
```cmd
cd swift_f0_cpp
build.bat
```

#### Option B: Manual build with CMake
```cmd
cd swift_f0_cpp
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Running the Application

After building, the executable will be in `build/Release/`:

```cmd
cd build\Release
swift_f0.exe recorded_samples.wav
```

### Command Line Arguments

```cmd
swift_f0.exe [audio_file] [fmin] [fmax] [confidence_threshold]
```

Examples:
```cmd
# Default parameters (same as Python example)
swift_f0.exe recorded_samples.wav

# Custom frequency range for speech
swift_f0.exe speech.wav 65 400 0.9

# Custom parameters
swift_f0.exe music.wav 46.875 2093.75 0.85
```

## Output

The program will print:
- Processing information
- Total number of frames
- Percentage of voiced frames
- Statistics for voiced regions (min/max/average pitch and confidence)
- Detailed information for the first 10 frames

## Differences from Python Version

1. **Audio Loading**: Uses a simple WAV file reader instead of librosa
   - Supports 16-bit and 32-bit WAV files
   - Automatically converts stereo to mono

2. **Resampling**: Uses linear interpolation instead of librosa's resampling
   - For better quality, consider using a proper resampling library

3. **Output Format**: Prints results to console instead of returning objects
   - You can modify `main.cpp` to export to CSV or other formats

## Troubleshooting

### "ONNX Runtime not found"
- Ensure the NuGet package is extracted to the correct location
- Check that the version number in `CMakeLists.txt` matches your downloaded version

### "Model file not found"
- Copy `model.onnx` from the Python package to the project directory

### Build errors with Visual Studio
- Make sure you have the C++ development workload installed
- Try using a different generator: `-G "Visual Studio 16 2019"` for VS 2019

### Runtime errors
- Ensure `onnxruntime.dll` is in the same directory as the executable
- Check that your WAV file is valid and readable

## Performance Notes

- The C++ version should be faster than Python for file I/O and array operations
- ONNX Runtime inference speed should be comparable
- Consider using multithreading for batch processing multiple files