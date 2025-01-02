import cv2
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import uuid
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import mediapipe as mp
from utility import *

# Initialize
app = FastAPI()
mtcnn = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                  max_num_faces=1, min_detection_confidence=0.5)

# Direction to store uploaded image
UPLOAD_DIR = "uploads"
# Direction to store processed image
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Upload file api
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type. Please upload an image (.jpeg or .png)."},
                            status_code=400)

    # ------------------------
    # Generate unique name
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"

    # Save the file to the upload directory
    file_location = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Convert image to RGB (if possible)
    img = Image.open(file_location)
    img = img.convert("RGB")

    # Crop image if user send a big picture

    # # Save the processed image
    # processed_location = os.path.join(PROCESSED_DIR, unique_name)
    # img.save(processed_location)

    # Detect face using MTCNN
    boxes, conf = mtcnn.detect(img)  # return boxes and confidence score

    if boxes is None or (conf is not None and conf.max() < 0.5):
        return JSONResponse(content={"error": "No face detected in the image. Please send a valid image"},
                            status_code=400)
    # Extract box info
    face_box = boxes[0]  # [x1,y1,x2,y2]
    # print(face_box)

    # # Crop the face from the image
    # x1, y1, x2, y2 = map(int, face_box)
    # cropped_face = img.crop((x1, y1, x2, y2))
    # # cropped_face = cropped_face.resize((256, 256))
    # cropped_face.save(os.path.join(PROCESSED_DIR, f"cropped_{unique_name}"))

    # Extend (WIP)

    # ------PREPROCESS IMAGE--------

    # Normalize pixel values (No need because MediaPipe need base value)
    img_array = np.array(img)

    # img_array = img_array.astype(np.uint8)
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    gray_img = cv2.equalizeHist(gray_img)
    # Denoise
    gray_img = cv2.bilateralFilter(gray_img, 9, 75, 75)

    # Reduce the glasses glare
    mask = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY)[1]  # Create a mask for bright spots
    preprocessed_img = cv2.inpaint(gray_img, mask, 3, cv2.INPAINT_TELEA)
    # # Debug
    # cv2.imshow('Landmarks', preprocessed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
    # --------END OF PRECESSING------------

    # Detect facial landmarks using MediaPipe
    results = face_mesh.process(preprocessed_img)

    if not results.multi_face_landmarks:
        return JSONResponse(content={"error": "No facial landmarks detected. Please try again"}, status_code=400)

    # Extract landmarks
    landmarks = []
    for landmark in results.multi_face_landmarks[0].landmark:
        x = int(landmark.x * img_array.shape[1])  # Scale with width
        y = int(landmark.y * img_array.shape[0])  # Scale with height
        landmarks.append((x, y))

    # CALCULATE EYES ASPECT RATIO
    left_ear = calculate_ear(landmarks)[0]
    right_ear = calculate_ear(landmarks)[1]

    # VALIDATE EYES OPEN
    threshold = 0.17
    eyes_open = left_ear > threshold and right_ear > threshold
    if not eyes_open:
        return JSONResponse(content={"Message": "Eyes are not open. Please upload a valid image."}, status_code=400)

    # Calculate the yaw, pitch, roll angles
    yaw, pitch, roll = head_pose_estimation(img_array, landmarks)

    ANGLE_THRESHOLD = 29
    if yaw > ANGLE_THRESHOLD \
            and abs(360 - yaw) > ANGLE_THRESHOLD \
            or abs(pitch) > ANGLE_THRESHOLD \
            or abs(roll) > ANGLE_THRESHOLD:
        return JSONResponse(content={"Message": "The face angles must be lower than 30 degrees. Please try again."},
                            status_code=400)
    # Test area
    #
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         # Visualize all landmarks
    #         annotated_image = visualize_landmarks(img, face_landmarks)
    #
    #         # Optionally save or display the annotated image
    #         cv2.imshow('Landmarks', annotated_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # # Validation logic (Add more in future)
    return JSONResponse(content={"Message": "Valid image."}, status_code=200)

    # For Debug only

    # return {
    #     "original_filename": file.filename,
    #     "face_box": face_box.tolist(),
    #     "landmarks": landmarks,  # List of (x, y) points for facial landmarks
    #     "landmarks_number": len(landmarks),
    #     "Message": "Eyes are open." if eyes_open else "Eyes are not open.",
    #     "EAR": [left_ear, right_ear],
    #     "yaw": yaw,
    #     "pitch": pitch,
    #     "roll": roll,
    #
    # }
