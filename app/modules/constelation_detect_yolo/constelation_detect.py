from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd

constelation_yolo_model = None

save_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + '..' + os.sep + '..'  + os.sep + 'temp' + os.sep

def load_model():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(current_folder + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep +'constellation_yolo8.pt')
    return model

def predict_image(image):
    global constelation_yolo_model, save_dir
    
    if constelation_yolo_model is None:
        print("Init model")
        constelation_yolo_model = load_model()
        
    # Realizează predicția
    results = constelation_yolo_model(image, save=True, save_dir=save_dir, project="temp", name='constelations_yolo')
    
    
    return process_result(results);
    
def process_result(results):
    
    
    result = results[0]
    
    class_names = result.names

    detected_constellations = []
    
    
    # Iterează prin boxurile detectate
    for box in result.boxes:
        class_id = int(box.cls[0])  # Obține ID-ul clasei
        confidence = box.conf[0]  # Obține scorul de încredere
        coords = box.xyxy[0].cpu().numpy()  # Coordonatele boxului (xmin, ymin, xmax, ymax)
        constellation_name = class_names[class_id]
        # Adaugă informațiile despre constelații detectate
        detected_constellations.append({
            "name": constellation_name,  # Folosește numele constelației
            "class_id": class_id,
            "confidence": float(confidence),
            "bounding_box": coords.tolist()  # Convertim în listă pentru a o putea trimite la template
            
        })
    
    # Convertim lista de dicționare într-un array structurat
    detected_constellations = np.array([
        (item["name"], item["class_id"], item["confidence"], item["bounding_box"])
        for item in detected_constellations
    ], dtype=[("name", "U20"), ("class_id", "i4"), ("confidence", "f4"), ("bounding_box", "4i4")])

    output_img_path = results[0].save_dir + os.sep  +results[0].path
    return detected_constellations, output_img_path;

if __name__=='__main__': # a simple test of this class
    img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/../../../test_img/star-2630050_1280.jpg')
    detected_constellations, output_img = predict_image(img)
    
    if (len(detected_constellations)>0):
        print("Constelație detectată!", detected_constellations);
        # Transformă lista de dicționare într-un DataFrame
        df = pd.DataFrame(detected_constellations[['name', 'confidence']])
        detected_constellations = df.groupby('name').aggregate({'confidence': 'max'}).sort_values(by='confidence').reset_index()
        print(detected_constellations.head())
        print("JSON:", detected_constellations)
        #img = cv2.imread(output_img);
        #cv2.imshow("Constelatii:", img)
        #cv2.waitKey(0)
    else:
        print("Nu a fost detectată nicio constelație.")