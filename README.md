# Face Detection Tools

A comprehensive Python package for high-quality face detection, extraction, and quality analysis from videos and images. This package provides GPU-accelerated face detection, advanced face quality metrics, and intelligent face selection algorithms.

## Features

### 1. Face Quality Analysis (`face_quality.py`)
- Comprehensive face quality metrics:
  - Blur detection (Laplacian and FFT methods)
  - Face alignment analysis
  - Face orientation detection (yaw, pitch, roll)
  - Eye openness detection
  - Brightness and contrast checking
  - Resolution assessment
- Configurable quality thresholds
- Detailed quality metrics output

### 2. Training Face Generation (`generateTrainingFaces.py`)
- GPU-accelerated face detection using MTCNN
- Intelligent face selection with quality filtering
- Multiprocessing support for parallel processing
- Advanced face tracking and uniqueness detection
- Progress tracking with ETA estimation
- Configurable sampling rates and batch sizes

## Installation

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Download the shape predictor model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### Dependencies
- tensorflow>=2.6.0
- mtcnn>=0.1.1
- dlib>=19.22.0
- opencv-python>=4.5.0
- numpy>=1.21.0
- face-recognition>=1.3.0
- scipy>=1.7.0
- tqdm>=4.62.0

## Usage

### Face Quality Analysis
```python
from faceDetectionTools import FaceQualityAnalyzer

# Initialize the analyzer
analyzer = FaceQualityAnalyzer()

# Analyze face quality
quality_score, metrics = analyzer.get_face_quality_score(face_image)

# Access detailed metrics
print(f"Quality Score: {quality_score}")
print("Detailed Metrics:", metrics)
```

### Training Face Generation

The face extraction tool now supports configuration-based usage for better flexibility and reproducibility.

#### Basic Usage

1. Using default configuration:
```bash
python generateTrainingFaces.py video.mp4
```

2. Using custom configuration:
```bash
python generateTrainingFaces.py video.mp4 --config custom_config.json
```

#### Configuration File

The tool uses a JSON configuration file (`faceGenConfig.json`) to control all aspects of face extraction. Here's the default configuration structure:

```json
{
    "output_dir": "extracted_faces",
    "max_faces": 30,
    "images_per_face": 30,
    "min_face_size": 40,
    "min_confidence": 0.95,
    "min_quality": 0.6,
    "batch_size": 4,
    "fps": 1.0,
    "use_gpu": true,
    "gpu_memory_fraction": 0.7,
    "face_similarity_threshold": 0.6,
    "skip_existing": true,
    "save_metadata": true,
    "quality_metrics": {
        "blur_threshold": 100,
        "brightness_range": [0.2, 0.8],
        "min_face_angle": 30,
        "min_eye_openness": 0.3
    },
    "logging": {
        "level": "INFO",
        "show_progress": true
    }
}
```

#### Configuration Parameters

1. **Basic Settings**
   - `output_dir`: Directory where extracted faces will be saved
   - `max_faces`: Maximum number of unique faces to extract
   - `images_per_face`: Number of images to save per unique face
   - `fps`: Frames per second to process from video

2. **Face Detection Settings**
   - `min_face_size`: Minimum face size in pixels
   - `min_confidence`: Minimum confidence score for face detection
   - `min_quality`: Minimum quality score for face selection
   - `face_similarity_threshold`: Threshold for determining unique faces

3. **Performance Settings**
   - `use_gpu`: Enable/disable GPU acceleration
   - `gpu_memory_fraction`: Fraction of GPU memory to use
   - `batch_size`: Batch size for processing

4. **Quality Metrics**
   - `blur_threshold`: Threshold for blur detection
   - `brightness_range`: Acceptable brightness range [min, max]
   - `min_face_angle`: Minimum acceptable face angle
   - `min_eye_openness`: Minimum eye aspect ratio

5. **Output Settings**
   - `skip_existing`: Skip processing if output directory exists
   - `save_metadata`: Save detailed metadata for each face

#### Output Structure

The tool creates the following directory structure:
```
output_dir/
├── face_1/
│   ├── frame_0001_quality_0.85.jpg
│   ├── frame_0015_quality_0.92.jpg
│   └── metadata.json
├── face_2/
│   ├── frame_0008_quality_0.88.jpg
│   ├── frame_0023_quality_0.90.jpg
│   └── metadata.json
└── extraction_stats.json
```

Each face directory contains:
- High-quality face images named with frame number and quality score
- `metadata.json` with detailed face metrics and extraction information
- Global `extraction_stats.json` with overall processing statistics

#### Custom Configuration Example

Create a custom configuration for high-quality face extraction:

```json
{
    "output_dir": "high_quality_faces",
    "max_faces": 50,
    "images_per_face": 50,
    "min_face_size": 60,
    "min_confidence": 0.98,
    "min_quality": 0.8,
    "batch_size": 8,
    "fps": 2.0,
    "quality_metrics": {
        "blur_threshold": 150,
        "brightness_range": [0.3, 0.7],
        "min_face_angle": 20,
        "min_eye_openness": 0.4
    }
}
```

This configuration will:
- Extract more faces with higher quality requirements
- Process frames at 2 FPS for more temporal coverage
- Use stricter quality metrics for better results
- Output to a custom directory

## Quality Metrics

The face quality analyzer provides the following metrics:

1. **Blur Score** (0-1)
   - Uses both Laplacian variance and FFT analysis
   - Higher score indicates sharper image

2. **Alignment Score** (0-1)
   - Measures face alignment based on eye positions
   - Perfect alignment = 1.0

3. **Orientation Scores**
   - Yaw (horizontal rotation)
   - Pitch (vertical rotation)
   - Roll (tilt)
   - Each ranges from 0-1, where 1 = frontal

4. **Eye Openness** (0-1)
   - Based on Eye Aspect Ratio (EAR)
   - Higher score indicates more open eyes

5. **Brightness & Contrast** (0-1)
   - Optimal brightness around 128/255
   - Higher contrast scores indicate better detail

## Performance Optimization

### GPU Acceleration
- Automatically detects and configures available GPUs
- Dynamic batch size adjustment based on GPU memory
- Configurable memory growth settings

### Parallel Processing
- Multiprocessing for face detection
- Thread pooling for I/O operations
- Efficient frame extraction and processing

## Error Handling

The package includes comprehensive error handling:
- Input validation for video files
- GPU memory management
- Progress tracking and ETA estimation
- Detailed error messages and logging

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Face Tools

A collection of tools for processing and evaluating facial images for AI training, with a focus on high-quality image selection and GPU acceleration.

## Features

- **Face Quality Assessment**: GPU-accelerated tool for evaluating facial images
- **Strict Quality Criteria**: Ensures only the best images are selected for training
- **Batch Processing**: Efficiently process multiple images
- **Detailed Analysis**: Quality scores and specific feedback for each image
- **Resource Monitoring**: Built-in system resource management

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- CUDA-compatible GPU (optional, but recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/faceTools.git
cd faceTools
```

2. Install dependencies:
```bash
pip install -r requirementsCheckFaces.txt
```

3. Download the face landmarks predictor model:
```bash
# Create a models directory
mkdir -p models

# Download the model file
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P models/

# Extract the file
bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2
```

Alternatively, you can:
- Download manually from [dlib's model files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract using any bzip2-compatible tool
- Place the extracted .dat file in the `models` directory

4. Install and start Ollama:
Visit [Ollama's installation guide](https://ollama.ai/download) for detailed instructions.

5. Pull the required vision model:
```bash
ollama pull llama3.2-vision:latest
```

## Usage

### Face Quality Assessment (checkFaces.py)

This tool evaluates facial images for AI training suitability using strict quality criteria.

```bash
python checkFaces.py "/path/to/image/directory" [--debug]
```

Options:
- `--debug`: Process only one image and show detailed output

The tool will:
1. Analyze each image for:
   - Face orientation (must be frontal)
   - Eye visibility (both eyes must be clear)
   - Lighting quality
   - Image resolution
   - Occlusions (hats, glasses, etc.)
   - Expression
2. Generate a CSV report with:
   - Quality assessment (Yes/No)
   - Detailed explanation
   - Quality score (1-10)
   - Image dimensions
   - File size
   - Processing time

Output is saved to `faceRatings/face_quality_results_[timestamp].csv`

### Quality Scoring System

Images are rated on a scale of 1-10:
- 10: Perfect training image - frontal, clear, well-lit, no issues
- 9: Excellent - very minor imperfections
- 8: Good - slight angle or lighting issues
- 7: Usable - noticeable but acceptable issues
- 1-6: Not suitable for training - multiple issues

Only images scoring 7 or higher are marked as suitable for training.

## Configuration

Default settings in `Config` class (`checkFaces.py`):
- `MAX_IMAGE_SIZE`: (1024, 1024)
- `BATCH_SIZE`: 5
- `TIMEOUT_SECONDS`: 30
- `MEMORY_THRESHOLD`: 90%
- `MODEL_NAME`: "llama3.2-vision:latest"

## Example Output

```csv
Folder,Image,Quality,Explanation,Score,Width,Height,File_Size_KB,Processing_Time_Sec
Player1,image1.jpg,No,"profile view, left eye not visible, wearing hat",2,1325,1715,1703.32,1.31
Player1,image2.jpg,Yes,"clear frontal view, good lighting, no occlusions",9,1200,1600,1500.45,1.25
```

## Error Handling

The tool includes:
- Comprehensive error logging
- System resource monitoring
- Timeout mechanisms
- Batch processing with recovery
- Invalid image detection

## Project Structure

```
faceTools/
├── checkFaces.py          # Main face quality assessment tool
├── face_quality.py        # Core quality assessment functions
├── findPlayerFaces.py     # Face detection utilities
├── generateTrainingFaces.py # Training data generation
├── models/               # Directory for model files
│   └── shape_predictor_68_face_landmarks.dat  # Face landmarks model
├── faceRatings/          # Output directory for CSV reports
├── requirementsCheckFaces.txt  # Python dependencies
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Include your license information here]
