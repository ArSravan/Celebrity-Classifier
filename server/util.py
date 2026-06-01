import os
import json
import base64
import joblib
import numpy as np
import cv2
import mediapipe as mp
import pywt

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------- GLOBALS --------------------
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

# -------------------- MODEL PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "saved_model.pkl")
CLASS_DICT_PATH = os.path.join(ARTIFACTS_DIR, "class_dictionary.json")

# -------------------- MEDIAPIPE SETUP --------------------
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=3
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = 33
RIGHT_EYE = 263

# -------------------- LOAD ARTIFACTS --------------------
def load_saved_artifacts():
    global __model, __class_name_to_number, __class_number_to_name

    print("loading saved artifacts...")

    with open(CLASS_DICT_PATH, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    __model = joblib.load(MODEL_PATH)

    print("loading saved artifacts...done")


# -------------------- BASE64 IMAGE --------------------
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# -------------------- WAVELET FEATURE --------------------
def w2d(img, mode='db1', level=2):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255

    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)

    return imArray_H


# -------------------- FACE CROPPING (MEDIA PIPE + ALIGNMENT) --------------------
def get_cropped_faces(image_path=None, image_base64_data=None):

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        return []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) != 1:
        return []

    h, w = img.shape[:2]
    faces = []

    for lm in result.face_landmarks:

        # ---------------- ALIGN FACE ----------------
        left_eye = lm[LEFT_EYE]
        right_eye = lm[RIGHT_EYE]

        x1 = left_eye.x * w
        y1 = left_eye.y * h
        x2 = right_eye.x * w
        y2 = right_eye.y * h

        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx))
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned_img = cv2.warpAffine(img, M, (w, h))

        # ---------------- RE-DETECT AFTER ALIGNMENT ----------------
        rgb2 = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)

        mp_image2 = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb2
        )

        result2 = detector.detect(mp_image2)

        if not result2.face_landmarks:
            continue

        lm2 = result2.face_landmarks[0]

        # ---------------- FACE BOUNDING BOX ----------------
        xs = [int(p.x * w) for p in lm2]
        ys = [int(p.y * h) for p in lm2]

        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))

        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        roi = aligned_img[y1:y2, x1:x2]

        if roi.shape[0] < 60 or roi.shape[1] < 60:
            continue

        faces.append(roi)

    return faces


# -------------------- CLASSIFY IMAGE --------------------
def classify_image(image_base64_data=None, file_path=None):

    faces = get_cropped_faces(file_path, image_base64_data)

    results = []

    for face in faces:

        # MUST MATCH TRAINING EXACTLY
        face_resized = cv2.resize(face, (64, 64))
        face_wavelet = w2d(face, 'db1', 2)
        face_wavelet_resized = cv2.resize(face_wavelet, (64, 64))

        feature_vector = np.concatenate((
            face_resized.flatten(),
            face_wavelet_resized.flatten()
        ))

        feature_vector = feature_vector.reshape(1, -1).astype(float)

        prediction = __model.predict(feature_vector)[0]
        probabilities = __model.predict_proba(feature_vector)[0]

        results.append({
            "class": __class_number_to_name[prediction],
            "class_probability": np.round(probabilities * 100, 2).tolist(),
            "class_dictionary": __class_name_to_number
        })

    return results


# -------------------- TEST --------------------
if __name__ == "__main__":
    load_saved_artifacts()

    print(classify_image(file_path="./test_images/virat3.jpg"))