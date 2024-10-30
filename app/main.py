from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import base64
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime

import app.modules.constelation_detect_yolo.constelation_detect as yolo_model

root = os.path.dirname(os.path.abspath(__file__))


app = FastAPI()
app.mount("/static", StaticFiles(directory="app" + os.sep + "static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include endpoint-urile pentru fiecare proiect
# app.include_router(project1_routes.router, prefix="/project1")
# app.include_router(project2_routes.router, prefix="/project2")
# app.include_router(project3_routes.router, prefix="/project3")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/yolo", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})

# Ruta pentru procesarea imaginii încărcate
@app.post("/yolo-upload-image", response_class=HTMLResponse)
async def upload_image(request:Request, image: UploadFile = File(...)):
    global constelation_yolo_model
    # Citește imaginea ca fișier binar
    image_data = await image.read()

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    detected_constellations, output_img_path = yolo_model.predict_image(img)
    if detected_constellations.any:
        result = True
        img = cv2.imread(output_img_path)
        img = base64.b64encode(img).decode('utf-8')
        
        
        # Transformă lista de dicționare într-un DataFrame
        df = pd.DataFrame(detected_constellations[['name', 'confidence']])
        detected_constellations = df.groupby('name').aggregate({'confidence': 'max'}).sort_values(by='confidence').reindex()
    else:
        result = False
        img = False;

    response = {
        "request": request,
        "result": result,
        "image_base64": img,
        "detected_constellations": detected_constellations.to_dict(orient="records")  
    }
    print("response:")
    print(detected_constellations)
    return templates.TemplateResponse("upload_image.html", response)
