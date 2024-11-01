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
CONSTELLATION_NAMES = [
    'aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia',
    'cygnus', 'gemini', 'leo', 'lyra', 'moon',
    'orion', 'pleiades', 'sagittarius', 'scorpius', 'taurus', 'ursa_major'
]
 
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
 
 
# Load the trained model
current_folder = os.path.dirname(os.path.abspath(__file__))    
model_filename = current_folder + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep +'constellation_rnn_model_noje_0.9083.pth'
print("current folder:", current_folder)
model = ConstellationRNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_filename, weights_only=True))
model.eval()
 
def load_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {file_path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
 
def parse_label(label_path):
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
 
def predict(image_path, label_path):
    # Load image
    image = load_image(image_path)
 
    # Load actual label
    actual_label = parse_label(label_path)
 
    # Prepare label input for LSTM
    if actual_label is not None and actual_label.size(0) > 0:
        label_input = actual_label.unsqueeze(0)  # Add batch dimension
    else:
        label_input = torch.zeros(1, 1, 5)  # Create a zero tensor with shape (1, 1, 5)
 
    # Make prediction
    with torch.no_grad():
        outputs = model(image, label_input)  # Pass the prepared label input
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
 
    # Get the predicted class index and corresponding probabilities
    predicted_class_id = torch.argmax(probabilities).item()
    predicted_constellation_name = CONSTELLATION_NAMES[predicted_class_id] if predicted_class_id < NUM_CLASSES else None
 
    # Get the actual constellation name
    actual_constellation_name = CONSTELLATION_NAMES[int(actual_label[0, 0].item())] if actual_label is not None and actual_label.size(0) > 0 else None
 
    # Format confidence scores as percentages
    confidence_scores = {name: f"{round(probabilities[0, i].item() * 100)}%" for i, name in enumerate(CONSTELLATION_NAMES)}  # Map constellation names to their probabilities
 
    # Collect results
    result = {
        'Image': os.path.basename(image_path),
        'Predicted Constellation': predicted_constellation_name,
        'Actual Constellation': actual_constellation_name,
        'Confidence Scores': confidence_scores  # Use the formatted confidence scores
    }
    return result
 
# Specify the directories
current_folder = os.path.dirname(os.path.abspath(__file__))    
images_dir = current_folder + os.sep + '..' +  os.sep + '..' + os.sep + '..' + os.sep + "__Constellation.v1i.yolov8" + os.sep + "test" + os.sep + "images"
labels_dir = current_folder + os.sep + '..' +  os.sep + '..' + os.sep + '..' + os.sep +  "__Constellation.v1i.yolov8" + os.sep + "test" + os.sep + "labels"
 
results = []
 
# Loop through all images in the images directory
for image_file in os.listdir(images_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
 
        # Check if corresponding label file exists
        if os.path.exists(label_path):
            result = predict(image_path, label_path)
            results.append(result)
        else:
            print(f"Label file not found for: {image_file}")
 
# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(current_folder + os.sep + "predictions_results.csv", index=False)
print("Results saved to predictions_results.csv")