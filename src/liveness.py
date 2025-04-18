
import sys
import cv2
import numpy as np
import base64
from ultralytics import YOLO

model = YOLO("\models\best_Latest_99.pt")
classNames = ["fake", "real"]
confidence_threshold = 0.80
def check_liveness(face):
    try:
        results = model(face, imgsz=640, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = classNames[cls].upper()

                if conf > confidence_threshold:
                    if name == "REAL":
                        return {
                            "success": True,
                            "liveness": True,
                            "face_detected": True,
                            "message":"Detected face is Real."
                        }
                    elif name == "FAKE":
                        return {
                            "success": False,
                            "liveness": False,
                            "face_detected": True,
                            "message":"Detected face is Fake."
                        }

        # If no boxes matched confidence threshold
        return {
            "success": False,
            "liveness": False,
            "face_detected": False,
            "message":"No Face Detected."
        }

    except Exception:
        return {
            "success": False,
            "liveness": False,
            "face_detected": False,
            "message":"No Face Detected."
        }



def check_liveness_register(attachments):
    try:
        for idx, attach in enumerate(attachments):
            try:
                img = attach.base64String
                img_data = base64.b64decode(img)
                nparr = np.frombuffer(img_data, np.uint8)
                face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                results = model(face, imgsz=640, verbose=False)
                face_found_in_frame = False

                for r in results:
                    for box in r.boxes:
                        face_found_in_frame = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = classNames[cls].upper()

                        if conf > confidence_threshold:
                            if name == "FAKE":
                                return {
                                    "success": False,
                                    "liveness": False,
                                    "face_detected": True,
                                    "message": f"Detected face is Fake in image {idx + 1}."
                                }

                if not face_found_in_frame:
                    return {
                        "success": False,
                        "liveness": False,
                        "face_detected": False,
                        "message": f"No Face Detected in image {idx + 1}."
                    }

            except Exception as img_error:
                return {
                    "success": False,
                    "liveness": False,
                    "face_detected": False,
                    "message": f"Error processing image {idx + 1}: {img_error}"
                }

        # If loop completes: all images had real faces and none were fake
        return {
            "success": True,
            "liveness": True,
            "face_detected": True,
            "message": "Detected face is Real in all images."
        }

    except Exception as e:
        return {
            "success": False,
            "liveness": False,
            "face_detected": False,
            "message": f"Unexpected error during liveness check: {str(e)}"
        }


