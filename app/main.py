"""
Script: main.py
Autor: Covaci Silviu

Descriere: Acest cod este scriptul principal al aplicatie care inițializează și pornește serverul web FastAPI, 
ascultând cererile API și gestionând rutele definite în aplicație.

Toate cele patru modele suportate de aplicație, precum și sistemul expert, sunt inițializate la pornire și 
apelează inferența la nevoie. Fiecare model utilizează un format standardizat pentru input și output, asigurând modularitatea și abstractizarea aplicației
"""

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import cv2
import os
import mimetypes
import base64
import pandas as pd
import numpy as np


from datetime import datetime

import app.modules.constelation_detect_yolo.constelation_detect as yolo_model
import app.modules.constelation_detect_cnn.constelation_detect_cnn as cnn_model
import app.modules.constelation_detect_rnn.constelation_detect_rnn_v2 as rnn_model
import app.modules.swin_transformer.constellation_swin_model as swin_model


import app.modules.system_expert.system_expert as system_expert_lib

root = os.path.dirname(os.path.abspath(__file__))


system_expert = None
app = FastAPI()
app.mount("/static", StaticFiles(directory="app" + os.sep + "static"), name="static")
templates = Jinja2Templates(directory="app/templates")



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/compare-upload-image", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})

# Ruta pentru procesarea imaginii încărcate
@app.post("/compare-upload-image", response_class=HTMLResponse)
async def upload_image(request:Request, image: UploadFile = File(...)):
    # Citește imaginea ca fișier binar
    image_data = await image.read()
    image_name = image.filename
    print("image name=", image_name)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = {}

    detected_constellations, output_img_path = yolo_model.predict_image(img)
    results['yolo'] = get_response_for_detected_constelations(detected_constellations=detected_constellations, output_img_path=output_img_path)
    
    detected_constellations, output_img_path = cnn_model.predict_image(img)
    results['cnn'] = get_response_for_detected_constelations(detected_constellations=detected_constellations, output_img_path=output_img_path)
    
    detected_constellations, output_img_path = swin_model.predict_image(img)
    results['swin'] = get_response_for_detected_constelations(detected_constellations=detected_constellations, output_img_path=output_img_path)

    detected_constellations, output_img_path = rnn_model.predict_image(img, image_name)
    results['rnn'] = get_response_for_detected_constelations(detected_constellations=detected_constellations, output_img_path=output_img_path)


    response = {
        "request": request,
        "results": results,
    }
    #print("response:", response)
    return templates.TemplateResponse("upload_image.html", response)

@app.get("/system-expert", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("system_expert.html", {"request": request})

@app.post("/system-expert/question", response_class=JSONResponse)
async def read_root(request: Request):
    
    params = await request.json()  # Extrage toți parametrii POST

    
    print("params=", params)
    answer = params.get("answer")
    constellation = params.get("constellation")
    question = params.get("question")
    
    print("params:", answer, constellation, question);
    
    answers = {'yes': 'yes', 'no':'no', 'dontKnow':"i don't know"}
               
    global system_expert
    if (system_expert is None):
        system_expert = system_expert_lib.Expert()
        system_expert.start()
    
    if (answer is None or answer == '0' or answer == 0):
        print("start system!!!");
        system_expert.start()
    else:
        
        answer = answers[answer]
        if constellation:
            print(f"process answer {answer} for constelation {constellation}");    
            system_expert.process_constellation_specific_anwer(constellation, answer)
        else:
            print(f"process answer {answer} for question {question}");    
            system_expert.process_answer(question, answer)
        
    data = system_expert.get_question()
    return JSONResponse(content={
        "response": data
    })
    
    
def get_response_for_detected_constelations(detected_constellations, output_img_path):
    if detected_constellations.any:
        result = True
        
        if (output_img_path):
            # Detectează tipul MIME în funcție de extensia fișierului
            mime_type, _ = mimetypes.guess_type(output_img_path)

            # Citirea și codificarea imaginii în base64 cu prefixul MIME corect
            with open(output_img_path, "rb") as image_file:
                img = image_file.read()
            
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            encoded_image = f"data:{mime_type};base64," + base64.b64encode(img).decode('utf-8')
        else:
            encoded_image = False
            
        df = pd.DataFrame(detected_constellations[['name', 'confidence']])
        
        detected_constellations = df.groupby('name').aggregate({'confidence': 'max'}).sort_values(by='confidence', ascending=False).reset_index()
        detected_constellations['confidence'] = (detected_constellations['confidence'].round(2)*100).round(0).astype('uint8')

    else:
        result = False
        encoded_image = False

    response = {
        "result": result,
        "image_base64": encoded_image,
        "detected_constellations": detected_constellations.to_dict(orient="records")  
    }
    return response