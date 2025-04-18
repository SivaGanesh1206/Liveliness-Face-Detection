

import torch
import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import albumentations as A

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Face Processing")

# Load YOLO model for face detection
model = YOLO("\models\yolov8n-face-lindevs.pt")

def detect_and_crop_faces(image_path):
    """Detect, crop, and resize faces from an image using YOLO-face."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return []

    results = model(image)
    cropped_faces = []
    
    for result in results:
        # resized_face = cv2.resize(result, (640, 480))
        # if result.shape[0]
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face = image[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                resized_face = cv2.resize(face, (112,112))
                cropped_faces.append(resized_face)

    logger.info(f"Detected {len(cropped_faces)} faces in {image_path}")
    return cropped_faces


def detect_cropped_face(image):
    if image is None:
        logger.error(f"Failed to load image: {image}")
        return []

    results = model(image)
    cropped_faces = []
    logger.info(f"Image Count {len(image)} ")
    logger.info(f"Result Count {len(results)} ")    

    
    for result in results:
        # resized_face = cv2.resize(result, (640, 480))
        # if result.shape[0]
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face = image[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                resized_face = cv2.resize(face, (112,112))
                cropped_faces.append(resized_face)

    # logger.info(f"Detected {len(cropped_faces)} faces in {cropped_faces}")
    return cropped_faces

def apply_augmentation(image):
    """Apply augmentations while preserving face integrity."""
    augmentations = [
        
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=20, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, p=0.5)
    ]
    
    augmented_images = []
    for aug in augmentations:
        augmented_images.append(aug(image=image)['image'])

    return augmented_images

def process_and_crop_faces(captured_images_dir, user_name):
    """Detect faces, apply augmentations, and extract embeddings using ArcFace."""
    embeddings = []
    labels = []
    total_images_processed = 0
    total_faces_detected = 0
    total_augmented_images = 0
    total_embeddings_collected = 0

    for image_name in os.listdir(captured_images_dir):
        image_path = os.path.join(captured_images_dir, image_name)
        if not image_path.endswith('.jpg'):
            continue

        faces = detect_and_crop_faces(image_path)
        if not faces:
            logger.warning(f"No faces detected in {image_path}")
            continue

        total_images_processed += 1
        total_faces_detected += len(faces)

        for face in faces:
            try:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                original_embedding = DeepFace.represent(face_rgb, model_name="ArcFace", detector_backend="skip", enforce_detection=False)[0]["embedding"]
                embeddings.append(original_embedding)
                labels.append(user_name)
                total_embeddings_collected += 1
            except Exception as e:
                logger.error(f"Error generating embedding for {image_path}: {e}")

            # Augment and extract embeddings
            augmented_faces = apply_augmentation(face)
            total_augmented_images += len(augmented_faces)

            for aug_face in augmented_faces:
                try:
                    aug_face_rgb = cv2.cvtColor(aug_face, cv2.COLOR_BGR2RGB)
                    embedding = DeepFace.represent(aug_face_rgb, model_name="ArcFace", detector_backend="skip", enforce_detection=False)[0]["embedding"]
                    embeddings.append(embedding)
                    labels.append(user_name)
                    total_embeddings_collected += 1
                except Exception as e:
                    logger.error(f"Error generating embedding for augmented face: {e}")

    # Log final statistics
    logger.info(f"Total images processed: {total_images_processed}")
    logger.info(f"Total faces detected: {total_faces_detected}")
    logger.info(f"Total augmented images: {total_augmented_images}")
    logger.info(f"Total embeddings collected: {total_embeddings_collected}")

    return embeddings, labels

