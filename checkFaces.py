#!/usr/bin/env python3

import os
import sys
import json
import csv
from pathlib import Path
from PIL import Image
from datetime import datetime
import argparse
from tqdm import tqdm
import ollama
import signal
import time
import psutil
from contextlib import contextmanager
import logging
import shutil

class Config:
    """Configuration class to hold all settings"""
    def __init__(self):
        self.MAX_IMAGE_SIZE = (1024, 1024)
        self.BATCH_SIZE = 5
        self.TIMEOUT_SECONDS = 30
        self.MEMORY_THRESHOLD = 90  # Percentage
        self.MODEL_NAME = "llama3.2-vision:latest"  # Using the correct vision model
        self.OUTPUT_DIR = "faceRatings"
        self.CSV_COLUMNS = ["Person", "Image", "Quality", "Explanation", "Score", "Width", "Height", "File_Size_KB", "Processing_Time_Sec"]

# Global configuration object
config = Config()

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations"""
    def signal_handler(signum, frame):
        raise TimeoutError("Request timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def check_system_resources():
    """Check system resources before processing"""
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.percent > config.MEMORY_THRESHOLD:
            print(f"Warning: System memory usage is at {memory.percent}%!")
            return False
            
        # Check GPU memory if possible
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_usage = (info.used / info.total) * 100
            if gpu_usage > config.MEMORY_THRESHOLD:
                print(f"Warning: GPU memory usage is at {gpu_usage:.1f}%!")
                return False
            print(f"GPU Memory - Used: {info.used/1024**3:.1f}GB, Free: {info.free/1024**3:.1f}GB, Total: {info.total/1024**3:.1f}GB")
        except Exception as e:
            print(f"Note: Unable to check GPU memory: {str(e)}")
            
        return True
    except Exception as e:
        print(f"Warning: Error checking system resources: {str(e)}")
        return True  # Continue if resource check fails

def is_valid_image_file(file_path: Path) -> bool:
    """
    Check if a file is a valid image file.
    Skip files that:
    - Start with ._ (macOS resource forks)
    - Are hidden files
    - Have invalid extensions
    """
    name = file_path.name
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return (
        file_path.is_file()
        and not name.startswith('._')
        and not name.startswith('.')
        and file_path.suffix.lower() in {ext.lower() for ext in valid_extensions}
    )

def check_ollama_status():
    """Check if Ollama is running and properly initialized."""
    try:
        print("Checking Ollama status...")
        with timeout(10):  # 10 second timeout for status check
            models = ollama.list()
            print(f"Available models: {models}")
            
            # Check if our required model is available
            model_available = any(model.model == config.MODEL_NAME for model in models.models)
            if not model_available:
                print(f"Warning: {config.MODEL_NAME} not found in available models!")
                return False
            return True
    except TimeoutError:
        print("Error: Ollama status check timed out")
        return False
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return False

def get_image_info(image_path):
    """Get image dimensions and file size."""
    try:
        with Image.open(image_path) as img:
            dimensions = img.size
        file_size = os.path.getsize(image_path) / 1024  # Convert to KB
        return dimensions, file_size
    except Exception as e:
        print(f"Error getting image info: {str(e)}")
        return None, None

def extract_json_from_text(text):
    """Extract JSON from text that might contain other content."""
    try:
        # First try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON-like content within the text
        try:
            # Look for content between curly braces
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If no valid JSON found, create a structured response from the text
        return {
            "suitable": "No" if any(negative in text.lower() for negative in ["blur", "dark", "occlu", "poor", "low"]) else "Yes",
            "explanation": text.strip(),
            "quality_score": 1  # Default quality score when we have to parse non-JSON response
        }

def process_single_image(image_path):
    """Send image to Ollama vision model and get quality assessment."""
    start_time = time.time()
    dimensions = None
    file_size = None
    
    try:
        # Get image dimensions and file size
        with Image.open(image_path) as img:
            dimensions = img.size
            file_size = os.path.getsize(image_path) / 1024  # Convert to KB
            
        # Process with Ollama
        prompt = """You are a VERY strict face quality assessment expert. Your job is to be extremely critical and selective.
Your primary goal is to find issues with facial images for AI training. Be harsh in your assessment - only the absolute best images should pass.

STRICT REQUIREMENTS for a "Yes":
1. Face MUST be clearly facing the camera (frontal view)
2. BOTH eyes must be clearly visible
3. NO occlusions (hats, sunglasses, masks, hands, etc.)
4. Face must be well-lit and in sharp focus
5. Neutral or natural expression preferred
6. High enough resolution
7. No extreme shadows or backlighting

AUTOMATIC REJECTION if any of these are present:
- Profile or side view
- Single eye visible
- Face partially covered
- Blurry or low resolution
- Extreme expressions
- Heavy shadows or poor lighting
- Any occlusions (hats, glasses, etc.)

QUALITY SCORE (1-10):
10: Perfect training image - frontal, clear, well-lit, no issues
9: Excellent - very minor imperfections
8: Good - slight angle or lighting issues
7: Usable - noticeable but acceptable issues
1-6: Not suitable for training - multiple issues

You must respond in this exact JSON format:
{
    "suitable": "Yes/No",
    "explanation": "List key issues or qualities",
    "quality_score": X
}

Requirements:
- "suitable" must be exactly "Yes" or "No" (be very strict, reject if in doubt)
- "explanation" should be a single line of specific issues or qualities
- "quality_score" must be 1-10 (only 7+ should be marked as suitable)

Example responses:
{
    "suitable": "Yes",
    "explanation": "perfect frontal view, both eyes clear, well-lit, sharp, no occlusions",
    "quality_score": 10
}

{
    "suitable": "No",
    "explanation": "side angle, right eye not visible, wearing cap",
    "quality_score": 3
}

{
    "suitable": "No",
    "explanation": "blurry, shadows on face, looking down",
    "quality_score": 2
}

Remember: Your job is to be extremely critical. When in doubt, reject the image. We only want the absolute best quality images for training."""

        try:
            print("Sending request to Ollama...")  # Debug log
            with timeout(config.TIMEOUT_SECONDS):
                response = ollama.chat(
                    model=config.MODEL_NAME,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }]
                )
                print("Received response from Ollama")  # Debug log
                print(f"Raw response: {response}")  # Debug log
                
            try:
                content = response['message']['content']
                print(f"Attempting to parse response content: {content}")  # Debug log
                
                # Use the new JSON extraction function
                assessment = extract_json_from_text(content)
                print(f"Successfully parsed response: {assessment}")  # Debug log
                
                # Validate and clean up the response
                if not isinstance(assessment.get('suitable'), str):
                    assessment['suitable'] = str(assessment.get('suitable', 'Error'))
                if not isinstance(assessment.get('quality_score'), (int, float)):
                    assessment['quality_score'] = 1
                
                # Add additional metrics
                processing_time = time.time() - start_time
                assessment['processing_time'] = round(processing_time, 2)
                if dimensions:
                    assessment['width'] = dimensions[0]
                    assessment['height'] = dimensions[1]
                if file_size:
                    assessment['file_size_kb'] = round(file_size, 2)
                
                return {
                    "Image": os.path.basename(image_path),
                    "Quality": assessment.get('suitable', 'Error'),
                    "Explanation": assessment.get('explanation', 'Unknown error'),
                    "Score": assessment.get('quality_score', 1),
                    "Width": dimensions[0] if dimensions else 0,
                    "Height": dimensions[1] if dimensions else 0,
                    "File_Size_KB": round(file_size, 2) if file_size else 0,
                    "Processing_Time_Sec": f"{time.time() - start_time:.2f}"
                }
                
            except Exception as e:
                print(f"Error parsing response: {str(e)}")  # Debug log
                return {
                    "Image": os.path.basename(image_path),
                    "Quality": "Error",
                    "Explanation": f"Error parsing model response: {str(e)}",
                    "Score": 0,
                    "Width": 0,
                    "Height": 0,
                    "File_Size_KB": 0,
                    "Processing_Time_Sec": 0
                }
            
        except TimeoutError:
            print("Request timed out")  # Debug log
            return {
                "Image": os.path.basename(image_path),
                "Quality": "Error",
                "Explanation": f"Request timed out after {config.TIMEOUT_SECONDS} seconds",
                "Score": 0,
                "Width": 0,
                "Height": 0,
                "File_Size_KB": 0,
                "Processing_Time_Sec": 0
            }
        except Exception as e:
            print(f"Error during Ollama API call: {str(e)}")  # Debug log
            return {
                "Image": os.path.basename(image_path),
                "Quality": "Error",
                "Explanation": f"Error calling Ollama API: {str(e)}",
                "Score": 0,
                "Width": 0,
                "Height": 0,
                "File_Size_KB": 0,
                "Processing_Time_Sec": 0
            }
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Debug log
        return {
            "Image": os.path.basename(image_path),
            "Quality": "Error",
            "Explanation": f"Error processing image: {str(e)}",
            "Score": 0,
            "Width": 0,
            "Height": 0,
            "File_Size_KB": 0,
            "Processing_Time_Sec": 0
        }

def should_process_directory(dir_path: Path) -> bool:
    """
    Check if a directory should be processed.
    Skip directories that:
    - Start with ._ (macOS resource forks)
    - Start with . (hidden directories)
    - Are system directories
    """
    name = dir_path.name
    return (
        dir_path.is_dir()
        and not name.startswith('._')
        and not name.startswith('.')
        and not name in {'__pycache__', '$RECYCLE.BIN', 'System Volume Information'}
    )

def setup_destination_structure(source_dir: Path, dest_dir: Path) -> None:
    """
    Create a mirror of the source directory structure in the destination directory.
    Only creates directories, doesn't copy any files.
    """
    # Create the destination root if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through source directory and recreate structure
    for dir_path in source_dir.glob('**/'):
        if should_process_directory(dir_path):
            # Calculate relative path from source root
            rel_path = dir_path.relative_to(source_dir)
            # Create same directory in destination
            (dest_dir / rel_path).mkdir(parents=True, exist_ok=True)

def move_low_quality_image(image_path: Path, source_root: Path, dest_root: Path) -> None:
    """
    Move an image to the corresponding directory in the destination structure.
    Maintains the same relative path structure as the source.
    """
    # Calculate relative path from source root
    rel_path = image_path.relative_to(source_root)
    # Construct destination path
    dest_path = dest_root / rel_path
    
    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move the file
    try:
        shutil.move(str(image_path), str(dest_path))
        logging.info(f"Moved low-quality image: {rel_path}")
    except Exception as e:
        logging.error(f"Error moving file {image_path}: {str(e)}")

def process_directory(directory_path: str, debug: bool = False, move_threshold: float = None, dest_dir: str = None) -> None:
    """Process a directory containing person-specific subdirectories with face images."""
    directory_path = Path(directory_path)
    
    # Setup move functionality if specified
    if move_threshold is not None and dest_dir is not None:
        dest_dir = Path(dest_dir)
        setup_destination_structure(directory_path, dest_dir)
        logging.info(f"Created directory structure in {dest_dir}")
        logging.info(f"Will move images with score below {move_threshold} to {dest_dir}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"face_quality_results_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config.CSV_COLUMNS)
        writer.writeheader()
        
        # Get all valid person directories (skip hidden and system directories)
        person_dirs = [d for d in directory_path.iterdir() if should_process_directory(d)]
        
        if not person_dirs:
            logging.warning(f"No valid person directories found in {directory_path}")
            return
            
        logging.info(f"Found {len(person_dirs)} person directories to process")
        
        for person_dir in tqdm(person_dirs, desc="Processing people"):
            person_name = person_dir.name
            logging.info(f"Processing images for person: {person_name}")
            
            # Get all valid images in person's directory
            image_files = []
            for file_path in person_dir.iterdir():
                if is_valid_image_file(file_path):
                    image_files.append(file_path)
            
            if not image_files:
                logging.warning(f"No valid images found in directory for {person_name}")
                continue
                
            logging.info(f"Found {len(image_files)} valid images for {person_name}")
            
            if debug:
                # In debug mode, only process one image per person
                image_files = image_files[:1]
                logging.debug(f"Debug mode: Processing only first image for {person_name}")
            
            # Process images in batches
            for i in range(0, len(image_files), config.BATCH_SIZE):
                batch = image_files[i:i + config.BATCH_SIZE]
                
                try:
                    # Check system resources before processing batch
                    if not check_system_resources():
                        logging.error("System resources exceeded threshold. Stopping processing.")
                        return
                    
                    for image_path in batch:
                        try:
                            # Process single image with timeout
                            with timeout(config.TIMEOUT_SECONDS):
                                start_time = time.time()
                                result = process_single_image(str(image_path))
                                processing_time = time.time() - start_time
                                
                                # Add person information to result
                                result.update({
                                    'Person': person_name,
                                    'Processing_Time_Sec': f"{processing_time:.2f}"
                                })
                                
                                # Move image if score is below threshold
                                if move_threshold is not None and dest_dir is not None:
                                    score = float(result.get('Score', 0))
                                    if score < move_threshold:
                                        move_low_quality_image(image_path, directory_path, dest_dir)
                                
                                writer.writerow(result)
                                csvfile.flush()  # Ensure data is written immediately
                                
                        except TimeoutError:
                            error_result = {
                                'Person': person_name,
                                'Image': image_path.name,
                                'Quality': 'Error',
                                'Explanation': 'Processing timeout',
                                'Score': 0,
                                'Width': 0,
                                'Height': 0,
                                'File_Size_KB': 0,
                                'Processing_Time_Sec': config.TIMEOUT_SECONDS
                            }
                            writer.writerow(error_result)
                            
                            # Move error images if threshold is set
                            if move_threshold is not None and dest_dir is not None:
                                move_low_quality_image(image_path, directory_path, dest_dir)
                                
                        except Exception as e:
                            error_result = {
                                'Person': person_name,
                                'Image': image_path.name,
                                'Quality': 'Error',
                                'Explanation': f'Processing error: {str(e)}',
                                'Score': 0,
                                'Width': 0,
                                'Height': 0,
                                'File_Size_KB': 0,
                                'Processing_Time_Sec': 0
                            }
                            writer.writerow(error_result)
                            
                            # Move error images if threshold is set
                            if move_threshold is not None and dest_dir is not None:
                                move_low_quality_image(image_path, directory_path, dest_dir)
                
                except Exception as batch_error:
                    logging.error(f"Error processing batch: {str(batch_error)}")
                    continue
    
    logging.info(f"Processing complete. Results saved to: {csv_path}")
    print(f"\nProcessing complete. Results saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process face images for quality assessment.')
    parser.add_argument('directory', type=str, help='Directory containing person-specific subdirectories with face images')
    parser.add_argument('--debug', action='store_true', help='Process only one image per person and show detailed output')
    parser.add_argument('--move', nargs=2, metavar=('THRESHOLD', 'DESTINATION'),
                      help='Move images below THRESHOLD score to DESTINATION directory')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        move_threshold = float(args.move[0]) if args.move else None
        dest_dir = args.move[1] if args.move else None
        process_directory(args.directory, args.debug, move_threshold, dest_dir)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)