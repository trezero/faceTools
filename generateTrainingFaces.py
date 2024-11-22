import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from collections import defaultdict
import tensorflow as tf
from mtcnn import MTCNN
from typing import NamedTuple, List, Dict, Tuple, Optional
import subprocess
import os
import face_recognition
from scipy.spatial.distance import cosine
import multiprocessing
from tqdm import tqdm
import json
from dataclasses import dataclass, asdict
import logging

from .face_quality import FaceQualityAnalyzer

@dataclass
class FaceDetectionConfig:
    output_dir: str
    max_faces: int
    images_per_face: int
    min_face_size: int
    min_confidence: float
    min_quality_score: float
    batch_size: int
    frames_per_second: float
    use_gpu: bool
    gpu_memory_fraction: float
    face_similarity_threshold: float
    skip_existing: bool
    save_metadata: bool
    quality_metrics: Dict[str, Any]
    logging: Dict[str, Any]

    @classmethod
    def from_file(cls, config_path: str) -> 'FaceDetectionConfig':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def get_default_config(cls) -> 'FaceDetectionConfig':
        default_config_path = Path(__file__).parent / 'faceGenConfig.json'
        return cls.from_file(str(default_config_path))

def parse_args() -> Tuple[str, FaceDetectionConfig]:
    parser = argparse.ArgumentParser(description='Extract faces from video for training')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--config', type=str, help='Path to configuration file (optional)')
    
    args = parser.parse_args()
    
    if args.config:
        config = FaceDetectionConfig.from_file(args.config)
    else:
        config = FaceDetectionConfig.get_default_config()
    
    return args.video_path, config

class FaceOccurrence(NamedTuple):
    frame_num: int
    image: np.ndarray
    quality_score: float
    quality_metrics: dict
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h

def configure_gpu(memory_fraction: float = 1.0) -> bool:
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Using CPU only.")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 1024 * memory_fraction)]
            )
        print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s)")
        return True
    except RuntimeError as e:
        print(f"GPU configuration failed: {e}")
        return False

def get_optimal_batch_size(config: FaceDetectionConfig, has_gpu: bool) -> int:
    """Determine optimal batch size based on available GPU memory."""
    if not has_gpu:
        return 1

    try:
        gpu = tf.config.experimental.get_visible_devices('GPU')[0]
        gpu_memory = tf.config.experimental.get_memory_info(gpu)['free']
        return min(config.batch_size, max(1, int(gpu_memory / (1024 * 1024 * 100))))
    except:
        return config.batch_size

def process_video_info(video_path: str) -> Tuple[cv2.VideoCapture, int, int, float]:
    """Get video information and validate the video file."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    print(f"\nVideo Information:")
    print(f"Path: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration:.2f} seconds")

    return video, total_frames, fps, duration

def detect_faces_batch(detector: MTCNN, frames: List[np.ndarray], 
                      frame_numbers: List[int], config: FaceDetectionConfig,
                      quality_analyzer: FaceQualityAnalyzer) -> List[FaceOccurrence]:
    """Detect and analyze faces in a batch of frames."""
    face_occurrences = []
    
    # Process frames in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        batch_faces = pool.map(detector.detect_faces, frames)

    for i, faces in enumerate(batch_faces):
        frame = frames[i]
        frame_num = frame_numbers[i]
        
        for face in faces:
            if face['confidence'] < config.min_confidence:
                continue

            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            
            # Skip if face is too small
            if w < config.min_face_size or h < config.min_face_size:
                continue

            # Get comprehensive face quality metrics
            quality_score, quality_metrics = quality_analyzer.get_face_quality_score(face_img)
            if quality_score < config.min_quality_score:
                continue

            try:
                face_encoding = face_recognition.face_encodings(face_img)[0]
            except IndexError:
                continue

            face_occurrences.append(
                FaceOccurrence(
                    frame_num=frame_num,
                    image=face_img,
                    quality_score=quality_score,
                    quality_metrics=quality_metrics,
                    embedding=face_encoding,
                    bbox=face['box']
                )
            )

    return face_occurrences

def save_face_data(face_id: int, occurrences: List[FaceOccurrence], 
                  output_path: Path, images_per_face: int) -> None:
    """Save face images and metadata."""
    face_folder = output_path / f"face{face_id:02d}"
    face_folder.mkdir(exist_ok=True)

    # Sort occurrences by quality score
    occurrences.sort(key=lambda x: x.quality_score, reverse=True)

    # Save face metadata
    metadata = {
        "face_id": face_id,
        "total_occurrences": len(occurrences),
        "best_quality_score": occurrences[0].quality_score,
        "average_quality_score": np.mean([o.quality_score for o in occurrences]),
        "quality_metrics": occurrences[0].quality_metrics,
        "bbox_statistics": {
            "average_size": np.mean([o.bbox[2] * o.bbox[3] for o in occurrences]),
            "min_size": min([o.bbox[2] * o.bbox[3] for o in occurrences]),
            "max_size": max([o.bbox[2] * o.bbox[3] for o in occurrences])
        }
    }
    
    with open(face_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Select best quality samples with temporal distribution
    if len(occurrences) <= images_per_face:
        samples = occurrences
    else:
        # Divide occurrences into segments for temporal distribution
        segment_size = len(occurrences) // images_per_face
        samples = []
        for i in range(0, len(occurrences), segment_size):
            segment = occurrences[i:i + segment_size]
            samples.append(max(segment, key=lambda x: x.quality_score))
        samples = samples[:images_per_face]

    # Save selected samples with quality information
    for i, sample in enumerate(samples):
        output_file = face_folder / f"frame_{sample.frame_num:04d}_quality_{sample.quality_score:.2f}.jpg"
        cv2.imwrite(str(output_file), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))

def extract_faces(video_path: str, config: FaceDetectionConfig) -> None:
    """
    Extract high-quality face images from a video with advanced face detection and quality analysis.
    
    Args:
        video_path: Path to the input video file
        config: Configuration for face detection parameters
    """
    # Setup and validation
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    has_gpu = configure_gpu(config.gpu_memory_fraction)
    config.batch_size = get_optimal_batch_size(config, has_gpu)
    
    # Initialize components
    quality_analyzer = FaceQualityAnalyzer()
    detector = MTCNN(min_face_size=config.min_face_size)
    
    # Process video
    video, total_frames, fps, duration = process_video_info(video_path)
    frame_interval = max(1, int(fps / config.frames_per_second))
    
    # Initialize tracking variables
    face_occurrences: Dict[int, List[FaceOccurrence]] = defaultdict(list)
    known_face_encodings: List[np.ndarray] = []
    known_face_ids: List[int] = []
    face_count = 0
    processed_frames = 0
    
    print("\nFirst pass: Identifying unique faces...")
    start_time = time.time()
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            frames = []
            frame_numbers = []
            
            # Read batch of frames
            for _ in range(config.batch_size):
                for _ in range(frame_interval):
                    ret, frame = video.read()
                    if not ret:
                        break
                    processed_frames += 1
                    pbar.update(1)
                
                if not ret:
                    break
                    
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_numbers.append(processed_frames)

            if not frames:
                break

            # Process batch
            new_occurrences = detect_faces_batch(detector, frames, frame_numbers, config, quality_analyzer)

            # Group faces by identity
            for occurrence in new_occurrences:
                new_face = True
                if known_face_encodings:
                    face_distances = [cosine(occurrence.embedding, enc) for enc in known_face_encodings]
                    min_distance = min(face_distances)
                    if min_distance < config.face_similarity_threshold:
                        face_id = known_face_ids[face_distances.index(min_distance)]
                        new_face = False
                        face_occurrences[face_id].append(occurrence)

                if new_face and face_count < config.max_faces:
                    face_count += 1
                    known_face_encodings.append(occurrence.embedding)
                    known_face_ids.append(face_count)
                    face_occurrences[face_count].append(occurrence)

    video.release()
    print(f"\nFirst pass complete. Found {face_count} unique faces.")

    print("\nSecond pass: Saving face data...")
    for face_id, occurrences in tqdm(face_occurrences.items(), desc="Saving faces"):
        save_face_data(face_id, occurrences, output_path, config.images_per_face)

    print(f"\nProcessing complete!")
    print(f"Found {face_count} unique faces")
    print(f"Results saved to: {output_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

def main():
    video_path, config = parse_args()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, config.logging['level']))
    logger = logging.getLogger(__name__)
    
    # Configure GPU
    if config.use_gpu:
        configure_gpu(config.gpu_memory_fraction)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize face detector with config parameters
    face_detector = MTCNN(
        min_face_size=config.min_face_size,
        steps_threshold=[0.6, 0.7, config.min_confidence]
    )
    
    # Process video
    extract_faces(video_path, config)

if __name__ == '__main__':
    main()