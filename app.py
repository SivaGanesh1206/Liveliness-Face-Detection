
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request 
from fastapi import Request, BackgroundTasks

import shutil
import os
import uvicorn
import logging
import base64
import numpy as np
import cv2
import io

from pydantic import BaseModel
from deepface import DeepFace
from psycopg2.pool import SimpleConnectionPool
from ultralytics import YOLO
from typing import List
from PIL import Image


from src.utils import ImageRequest, get_user_directory, clean_up_directory,save_images
from src.data import create_database, save_embeddings_to_db,update_embedding_for_user
from src.embeddings_handler import calculate_mean_embedding, load_embeddings_from_db, predict_face_identity
from src.face_processing import process_and_crop_faces,detect_cropped_face
from src.liveness import check_liveness,check_liveness_register


import logging
from logging.handlers import RotatingFileHandler

# === Setup Logging BEFORE FastAPI app is created ===
os.makedirs("logs", exist_ok=True)

info_handler = RotatingFileHandler("logs/info.log", maxBytes=5_000_000, backupCount=5)
info_handler.setLevel(logging.INFO)

error_handler = RotatingFileHandler("logs/error.log", maxBytes=5_000_000, backupCount=5)
error_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)
app = FastAPI()

# Setup for Jinja2 template engine
templates = Jinja2Templates(directory="templates")  # Directory where index.html is located

# Allow all origins (for development, you can restrict it in production)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Static files (CSS, JS) configuration
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Face Recognition API")

UPLOAD_DIR = "captured_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#USE YOUR OWN HTML FILES IN TEMPLATES FOLDER 

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})  # Render the index.html
@app.get("/index")
async def go_to_index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Registration endpoint to handle base64 images
@app.post("/register_data")
async def main(request: ImageRequest):
    create_database()
    user_name = request.mobile_number
    user = request.user_name
    attach = request.attachments

    real = check_liveness_register(attach)
    
    if real['liveness']:
        if not user_name:
            logger.error("Error: Username cannot be empty!")
            return {"status": "error", "message": "Username is required"}

        output_dir = get_user_directory(user_name)
        logger.info(f"Images will be saved in: {output_dir}")
        

        for idx, attachment in enumerate(attach):
            image_data = base64.b64decode(attachment.base64String)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            img_path = os.path.join(output_dir, f"original_{idx+1}.jpg")
            img.save(img_path)
        # if not captured_images:
        #     logger.error("Error: No images captured.")
        #     return {"status": "error", "message": "No images captured"}

        embeddings, labels = process_and_crop_faces(output_dir, user_name)
        normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
        mean_embedding = calculate_mean_embedding(normalized_embeddings)


        update_embedding_for_user(mean_embedding,user_name)# DON'T USE THIS FUNCTION IF YOU ARE CREATING NEW DB AND STORING ALL VALUES NEWLY IN DB ALONG WITH EMBEDDINGS
        # USE THIS FUNCTION ONLY IF YOU WANT TO SAVE ONLY EMBEDDINGS IN A EMBEDDINGS COLUMN IN A SPECIFIC DATABASE


        save_embeddings_to_db(mean_embedding, user_name, user)
        clean_up_directory(output_dir)

        # logger.info(f"Captured {len(captured_images)} images for {user_name}.")
        logger.info(f"Generated {len(embeddings)} embeddings for {user_name}.")
        try:
            save_images("images", attach, mode="Reg", username=str(user_name))
        except Exception as e:
            logger.error(f"[save_images] Failed to save images for {user_name}: {e}")
            return real
        
        

        return {"status": "success", "message": f"Embeddings stored for user ID: {user}"}

    else:
        if real['face_detected']:
            save_images("fake", [request.attachments[0]], mode="Reg", username="Fake")
        return real

# Attachment model definition
class Attachment(BaseModel):
    attachmentFilename: str
    fileType: int
    base64String: str

# Images model to hold the list of attachments
class Images(BaseModel):
    attachments: List[Attachment]



@app.post("/verify_data")
async def live_verification(request: Images):
    db_config = {
        'dbname': 'dbname',
        'user': 'username',
        'password': '*****',
        'host': 'localhost',
        'port': 'port number'
    }

    attach = request.attachments[0].base64String  # Access the first attachment

    img_data = base64.b64decode(attach)
    nparr = np.frombuffer(img_data, np.uint8)
    face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    real = check_liveness(face) 
    #logger.info(f"face {face} ")
    #logger.info(f"face count {len(face)} ")
    logger.info(f"Real : {real['liveness']} ")

    if real['liveness']:
        
        stored_embeddings, stored_labels,stored_mobile_number = load_embeddings_from_db(db_config)
        if len(stored_embeddings) == 0:
            logger.warning("No embeddings found in database.")
            return {"status": "error", "message": "No data in database for verification", "liveness": real}
        cropped_face=detect_cropped_face(face)
        try:
            embedding = DeepFace.represent(cropped_face[0], model_name='ArcFace',  detector_backend="skip", enforce_detection=False)[0]["embedding"]
            #logger.info(f"Embeddings : {embedding} ")


            predicted_label = predict_face_identity(np.array(embedding), stored_embeddings, stored_labels,stored_mobile_number)
            logger.info(f"Predicted label : {predicted_label} ")
            # mobile_number = str(predicted_label[2])
            if predicted_label[0] == "unknown":
                save_images("unknown", [request.attachments[0]], mode="Ver", username="Unknown")
                logger.warning("Unknown face detected. Skipping image save.")
                return {"status": "success", "liveness": real,"predicted_label":predicted_label[0],"threshold":float(predicted_label[1])}
            else:
                save_images("images", [request.attachments[0]], mode="Ver", username=predicted_label[3])

            return {"status": "success", "liveness":real, "predicted_label": predicted_label[0],"threshold":float(predicted_label[1]),"Index":int(predicted_label[2]),"EmployeeID":str(predicted_label[3])}


        except Exception as e:
            return {"status": "errors", "message": str(e),'liveness':real}

    else:
        if real['face_detected']:
            save_images("fake", [request.attachments[0]], mode="Ver", username="Fake")
            logger.warning("Fake face detected. Image saved in 'fake'.")
        else:
            
            logger.warning("No face detected and liveness failed. No image saved")

        return {"status": "success", "liveness": real}

        # return {"status": "error" , "liveness":real,  "message": "Detected face is not live."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=0000, ssl_keyfile='key.pem', ssl_certfile='cert.pem')

# uvicorn app:app --host 0.0.0.0 --port 0000 --ssl-keyfile "C:\SSL\key.pem" --ssl-certfile "C:\SSL\cert.pem"








