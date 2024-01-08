from fastapi import FastAPI, File, UploadFile
import cv2
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import urllib
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import json
app = FastAPI()

# LOAD MODEL
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# LABEL
labels = [
    'trash_ano_botolplastik',
    'trash_ano_kaleng',
    'trash_ano_kresekplastik',
    'trash_org_sisamakanan',
    'trash_org_cangkangtelur',
    'trash_org_kertas',
]


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    image = np.asarray(image)[..., :3]
    return image

def preprocess_image(image):
    image_resized = cv2.resize(image, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5
    return input_data

def get_prediction(input_data):
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    return boxes, classes, scores

def get_output(boxes, classes, scores, image_height, image_weight):
    results = {}
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):

            ymin = int(max(1,(boxes[i][0] * image_height)))
            xmin = int(max(1,(boxes[i][1] * image_weight)))
            ymax = int(min(image_height,(boxes[i][2] * image_height)))
            xmax = int(min(image_weight,(boxes[i][3] * image_weight)))
            label = labels[int(classes[i])]
            
            object = {
                "label":label,
                "scores":int(round(scores[i],2) * 100),
                "xmin":xmin,
                "ymin":ymin,
                "xmax":xmax,
                "ymax":ymax
            }
            
            
            results[f"{i}"] = object
    return results

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png","JPG")
    if not extension:
        return "Image must be jpg or png format!"
    
    image = read_imagefile(await file.read())
    image_height, image_weight, _ = image.shape
    
    input_data = preprocess_image(image)
    
    boxes, classes, scores = get_prediction(input_data)
    
    output = get_output(boxes, classes, scores, image_height, image_weight)

    return {
        "detections": output
    }

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success",
        },
        "data": None
    }), 200