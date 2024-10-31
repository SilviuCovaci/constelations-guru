import torch
import cv2
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

# Constants
IMAGE_SIZE = 224  # Resize dimension for images
NUM_CLASSES = 16

# Define the ConstellationRNN class
class ConstellationRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=64):
        super(ConstellationRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(64 * 56 * 56 + hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, images, labels):
        img_features = self.conv(images)
        img_features = img_features.view(img_features.size(0), -1)
        _, (hn, _) = self.lstm(labels)
        seq_features = hn.squeeze(0)
        combined = torch.cat((img_features, seq_features), dim=1)
        x = self.fc1(combined)
        x = nn.ReLU()(x)
        output = self.fc2(x)
        return output

constelation_rnn_model = None

# Class names (same as in your data.yaml)
CONSTELLATION_NAMES = [
    'aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia',
    'cygnus', 'gemini', 'leo', 'lyra', 'moon',
    'orion', 'pleiades', 'sagittarius', 'scorpius', 'taurus', 'ursa_major'
]

save_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + '..' + os.sep + '..'  + os.sep + 'temp' + os.sep

def parse_label(label_path):
    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append([class_id, x_center, y_center, width, height])
        return torch.tensor(labels, dtype=torch.float32) if labels else None
    else:
        return None

def load_model():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = current_folder + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep +'constellation_rnn_model_noje_0.9083.pth';
    model = ConstellationRNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict_image(image, image_name):
    global constelation_rnn_model, save_dir
    
    if constelation_rnn_model is None:
        print("Init model RNN")
        constelation_rnn_model = load_model()
        
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    
    # Load actual label
    current_folder = os.path.dirname(os.path.abspath(__file__))
    label_name = image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    label_path = current_folder + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'rnn_labels' + os.sep + label_name
    actual_label = parse_label(label_path)

    # Prepare label input for LSTM
    if actual_label is not None and actual_label.size(0) > 0:
        label_input = actual_label.unsqueeze(0)  # Add batch dimension
    else:
        label_input = torch.zeros(1, 1, 5)  # Create a zero tensor with shape (1, 1, 5)


    # Make prediction
    with torch.no_grad():
        outputs = constelation_rnn_model(image, label_input)  # Pass the prepared label input
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        probabilities2 = torch.sigmoid(outputs)
        
    
 
    # # Get the predicted class index and corresponding probabilities
    # predicted_class_id = torch.argmax(probabilities).item()
    # predicted_constellation_name = CONSTELLATION_NAMES[predicted_class_id] if predicted_class_id < NUM_CLASSES else None
 
    # # Get the actual constellation name
    # actual_constellation_name = CONSTELLATION_NAMES[int(actual_label[0, 0].item())] if actual_label is not None and actual_label.size(0) > 0 else None
 
    # Format confidence scores as percentages
    #confidence_scores = {name: f"{round(probabilities[0, i].item() * 100)}%" for i, name in enumerate(CONSTELLATION_NAMES)}  #
    confidence_scores = { name: round(probabilities[0, i].item(),2) for i, name in enumerate(CONSTELLATION_NAMES)}  #
    #confidence_scores2 = { name: round(probabilities2[0, i].item() * 100) for i, name in enumerate(CONSTELLATION_NAMES)}  #
    print("confidence_scores1:", confidence_scores)
    #print("confidence_scores2:", confidence_scores2)
    
    threshold = 0.2
    index = 0
    detected_constellations = []
    for label_name, probability   in confidence_scores.items():
        
        if (probability >= threshold):
            # Adaugă informațiile despre constelații detectate
            detected_constellations.append({
                "name": label_name, 
                "class_id": index,
                "confidence": float(probability),
                "bounding_box": None
                
            })
        index = index+1
        
   
    detected_constellations = np.array([
        (item["name"], item["class_id"], item["confidence"], item["bounding_box"])
        for item in detected_constellations
    ], dtype=[("name", "U20"), ("class_id", "i4"), ("confidence", "f4"),  ("bounding_box", "O")])    
    
    return detected_constellations, False
    
if __name__=='__main__': # a simple test of this class

    img_name = '2022-01-26-00-00-00-n_png_jpg.rf.369511b8057da25ac72fcf0eef444fec.jpg'
    img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/../../../test_img/' + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_constellations, output_img = predict_image(img, img_name)
    
    if (len(detected_constellations)>0):
        print("Constelație detectată!", detected_constellations);
        # Transformă lista de dicționare într-un DataFrame
        df = pd.DataFrame(detected_constellations[['name', 'confidence']])
        detected_constellations = df.groupby('name').aggregate({'confidence': 'max'}).sort_values(by='confidence').reset_index()
        print(detected_constellations.head())
        print("JSON:", detected_constellations)

    else:
        print("Nu a fost detectată nicio constelație.")