
import base64
import os
from typing import List
import uuid
import logging
from PIL import Image
import io
from datetime import datetime

from pydantic import BaseModel


# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Face Embedding API")



class Attachment(BaseModel):
    attachmentFilename: str
    fileType: int
    base64String: str

# ImageRequest model to hold user details and attachments
class ImageRequest(BaseModel):
    user_name: str
    mobile_number: str
    attachments: List[Attachment]

def get_user_directory(user_name):
    """Create a unique directory for the user to save captured images."""
    user_name = str(user_name)
    base_dir = "/captured_images"
    output_dir = os.path.join(base_dir, user_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def compress_image_to_15kb(image: Image.Image) -> bytes:
    for quality in range(95, 10, -5):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        size_kb = buffer.tell() / 1024
        if size_kb <= 15:
            return buffer.getvalue()
    return buffer.getvalue()  # fallback

def compress_image_to_25kb(image: Image.Image) -> bytes:
    for quality in range(95, 10, -5):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        size_kb = buffer.tell() / 1024
        if size_kb <= 25:
            return buffer.getvalue()
    return buffer.getvalue()

def save_images(folder: str, attachments: list, mode: str = "Reg", username: str = "") -> list:
    saved_files = []
    username = str(username)
    # Base directory for storing captured images
    base_dir ="\captured_images"
    
    # Determine full directory path based on mode
    full_path = os.path.join(base_dir, username, mode)
    os.makedirs(full_path, exist_ok=True)
    os.chmod(full_path, 0o777)

    if mode == "Reg" and username not in ["Fake", "Unknown"]:
        for existing_file in os.listdir(full_path):
            file_path = os.path.join(full_path, existing_file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for idx, attachment in enumerate(attachments):
        image_data = base64.b64decode(attachment.base64String)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        if mode == "Reg":
            compressed_img = compress_image_to_25kb(img)
        else:
            compressed_img = compress_image_to_15kb(img)

        timestamp = datetime.now().strftime("%d.%m.%Y_%H%M%S")
        filename = f"{username}_{timestamp}_{idx+1}.jpg"  # ✅ Unique filename using idx
        save_path = os.path.join(full_path, filename)

        with open(save_path, "wb") as f:
            f.write(compressed_img)
        os.chmod(save_path, 0o777)
        saved_files.append(save_path)

    return saved_files



def clean_up_directory(captured_images_dir):
    """
    Clean up the directory by removing all captured images.
    """
    logger.info(f"Cleaning up the directory: {captured_images_dir}")
    for file_name in os.listdir(captured_images_dir):
        file_path = os.path.join(captured_images_dir, file_name)
        if file_path.endswith('.jpg'):
            try:
                os.remove(file_path)
                logger.info(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")



