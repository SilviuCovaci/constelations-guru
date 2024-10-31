import torch
import cv2
import numpy as np
import timm
from torchvision import transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os

constelation_swin_model = None
constelation_swin_transform = None

# Class names (same as in your data.yaml)
class_names = ['aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia', 'cygnus',
               'gemini', 'leo', 'lyra', 'moon', 'orion', 'pleiades', 'sagittarius',
               'scorpius', 'taurus', 'ursa_major']

transformers_save_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + '..' + os.sep + '..'  + os.sep + 'temp' + os.sep + 'swin' + os.sep

def load_model(device):
    # Define transformations for the input image (same as during training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Adjust as necessary
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    current_folder = os.path.dirname(os.path.abspath(__file__))
        
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=16)
    model.load_state_dict(torch.load(current_folder + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'constellation_swin_model.pth', weights_only=True))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return transform, model
    
# Function to perform inference and draw bounding boxes
def predict_image(image):
    global constelation_swin_model, constelation_swin_transform, transformers_save_dir, class_names
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if constelation_swin_model is None:
        print("Init model")
        constelation_swin_transform, constelation_swin_model = load_model(device)
        
    # Preprocess the image
    img_transformed = constelation_swin_transform(image).unsqueeze(0)  # Add batch dimension
    img_transformed = img_transformed.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = constelation_swin_model(img_transformed)
    
    detected_constellations = []
    #probabilities = F.softmax(outputs, dim=1)
    probabilities = torch.sigmoid(outputs)
    threshold = 0.5

    index = 0
    for probability, label_name  in zip(probabilities[0], class_names):
        
        if (probability >= threshold):
            # Adaugă informațiile despre constelații detectate
            detected_constellations.append({
                "name": label_name, 
                "class_id": index,
                "confidence": float(probability),
                "bounding_box": None
                
            })
        index = index+1
   
    print(detected_constellations)
    detected_constellations = np.array([
        (item["name"], item["class_id"], item["confidence"], item["bounding_box"])
        for item in detected_constellations
    ], dtype=[("name", "U20"), ("class_id", "i4"), ("confidence", "f4"),  ("bounding_box", "O")])    
    
    return detected_constellations, False
    
    


if __name__=='__main__': # a simple test of this class
    img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/../../../test_img/test__1.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_constellations, output_img = predict_image(img_rgb)
    exit()
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
        
