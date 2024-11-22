import cv2
import numpy as np
from typing import Tuple, Dict
import dlib
from scipy.spatial import distance

class FaceQualityAnalyzer:
    def __init__(self):
        # Initialize face landmark predictor
        self.face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
    def get_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Extract facial landmarks using dlib."""
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dets = detector(gray, 1)
        if len(dets) == 0:
            return None
        
        shape = self.face_predictor(gray, dets[0])
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def check_blur(self, image: np.ndarray) -> float:
        """
        Enhanced blur detection using multiple methods.
        Returns a score between 0 (blurry) and 1 (sharp).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance (detail detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # FFT for frequency analysis (motion blur detection)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
        frequency_score = np.mean(magnitude_spectrum)
        
        # Combine scores
        blur_score = min(1.0, (laplacian_var / 500 + frequency_score / 1000) / 2)
        return blur_score

    def check_alignment(self, landmarks: np.ndarray) -> float:
        """
        Check face alignment using eye positions.
        Returns a score between 0 (misaligned) and 1 (well-aligned).
        """
        if landmarks is None:
            return 0.0
            
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Convert angle to alignment score (0 degrees = perfect alignment)
        alignment_score = max(0, 1 - abs(angle) / 45)
        return alignment_score

    def check_face_orientation(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Estimate face orientation (yaw, pitch, roll).
        Returns scores between 0 (extreme angle) and 1 (frontal).
        """
        if landmarks is None:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            
        # Calculate face symmetry for yaw
        left_side = landmarks[0:8]
        right_side = landmarks[8:17]
        symmetry = abs(np.mean(left_side) - np.mean(right_side))
        yaw_score = max(0, 1 - symmetry / 50)
        
        # Estimate pitch using nose and chin positions
        nose_bridge = landmarks[27:31]
        chin = landmarks[8]
        pitch_angle = abs(np.mean(nose_bridge, axis=0)[1] - chin[1])
        pitch_score = max(0, 1 - pitch_angle / 50)
        
        # Roll score from alignment
        roll_score = self.check_alignment(landmarks)
        
        return {
            "yaw": yaw_score,
            "pitch": pitch_score,
            "roll": roll_score
        }

    def check_eye_openness(self, landmarks: np.ndarray) -> float:
        """
        Check if eyes are open using eye aspect ratio (EAR).
        Returns a score between 0 (closed) and 1 (open).
        """
        if landmarks is None:
            return 0.0
            
        def eye_aspect_ratio(eye_landmarks):
            vertical_dist1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
            vertical_dist2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
            horizontal_dist = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
            ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            return ear
            
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average EAR (typical threshold is 0.2)
        ear = (left_ear + right_ear) / 2
        return min(1.0, max(0.0, (ear - 0.2) / 0.15))

    def check_brightness_contrast(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Analyze image brightness and contrast.
        Returns scores between 0 (poor) and 1 (good).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Brightness score
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        
        # Contrast score
        std_dev = np.std(gray)
        contrast_score = min(1.0, std_dev / 64)
        
        return brightness_score, contrast_score

    def get_face_quality_score(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Comprehensive face quality assessment.
        Returns overall score and detailed metrics.
        """
        if image.size == 0:
            return 0.0, {}
            
        # Check minimum size
        height, width = image.shape[:2]
        if height < 64 or width < 64:
            return 0.0, {}
            
        # Get facial landmarks
        landmarks = self.get_landmarks(image)
        
        # Calculate individual metrics
        blur_score = self.check_blur(image)
        alignment_score = self.check_alignment(landmarks) if landmarks is not None else 0.0
        orientation_scores = self.check_face_orientation(landmarks)
        eye_score = self.check_eye_openness(landmarks) if landmarks is not None else 0.0
        brightness_score, contrast_score = self.check_brightness_contrast(image)
        
        # Calculate weighted average for final score
        weights = {
            'blur': 0.3,
            'alignment': 0.15,
            'orientation': 0.2,
            'eyes': 0.15,
            'brightness': 0.1,
            'contrast': 0.1
        }
        
        orientation_avg = np.mean(list(orientation_scores.values()))
        
        final_score = (
            weights['blur'] * blur_score +
            weights['alignment'] * alignment_score +
            weights['orientation'] * orientation_avg +
            weights['eyes'] * eye_score +
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score
        )
        
        metrics = {
            'blur_score': blur_score,
            'alignment_score': alignment_score,
            'orientation_scores': orientation_scores,
            'eye_openness': eye_score,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'final_score': final_score
        }
        
        return final_score, metrics
