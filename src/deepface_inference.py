# src/deepface_inference.py

from deepface import DeepFace

def generate_embedding(face_image_path):
    """Generate ArcFace embedding for a cropped face image."""
    return DeepFace.represent(face_image_path, model_name='ArcFace', detector_backend='retinaface')
