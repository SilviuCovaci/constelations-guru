import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import csv
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score
 
# Define constants
IMAGE_SIZE = 224
NUM_CLASSES = 16
 
# Map constellation indices to names
CONSTELLATION_NAMES = [
    'aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia',
    'cygnus', 'gemini', 'leo', 'lyra', 'moon', 'orion',
    'pleiades', 'sagittarius', 'scorpius', 'taurus', 'ursa_major'
]
 
 
def load_image(file_path):
    """Load and resize an image."""
    img = cv2.imread(file_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), os.path.basename(file_path)
 
 
def parse_label(label_path):
    """Parse YOLO format label file into a multi-hot vector."""
    multi_hot_labels = np.zeros(NUM_CLASSES, dtype=np.float32)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            if class_id < NUM_CLASSES:
                multi_hot_labels[class_id] = 1
    return torch.tensor(multi_hot_labels, dtype=torch.float32)
 
 
def load_data(image_path, label_path):
    """Load images and labels, return stacked tensors and file names."""
    images, labels, filenames = [], [], []
    image_files = glob(os.path.join(image_path, "*.jpg"))
    for img_path in image_files:
        label_file = os.path.join(label_path, os.path.basename(img_path).replace(".jpg", ".txt"))
        image, filename = load_image(img_path)
        label = parse_label(label_file)
        images.append(image)
        labels.append(label)
        filenames.append(filename)
    return torch.stack(images), torch.stack(labels), filenames
 
 
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
 
        self.lstm = nn.LSTM(input_size=num_classes, hidden_size=hidden_size, batch_first=True)
 
        img_feature_size = 64 * 56 * 56
        self.fc1 = nn.Linear(img_feature_size + hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
 
    def forward(self, images, labels):
        img_features = self.conv(images)
        img_features = img_features.view(img_features.size(0), -1)
 
        _, (hn, _) = self.lstm(labels.unsqueeze(1))
        seq_features = hn[-1]
 
        combined = torch.cat((img_features, seq_features), dim=1)
 
        x = self.fc1(combined)
        x = nn.ReLU()(x)
        output = self.fc2(x)
 
        return output
 
 
def evaluate_model(model, dataloader, filenames):
    model.eval()
    all_labels, all_predictions = [], []
    total_correct = 0
 
    # Open CSV file to write results
    with open("evaluation_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Image Name"] + CONSTELLATION_NAMES)
 
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                outputs = model(images, labels)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
 
                total_correct += (predicted == labels).sum().item()
 
                # Write each image's predictions to the CSV
                for i in range(images.size(0)):
                    img_name = filenames[batch_idx * dataloader.batch_size + i]
                    confidences = probabilities[i].cpu().numpy() * 100  # Convert to percentages
                    row = [img_name] + [f"{conf:.2f}" for conf in confidences]
                    writer.writerow(row)
 
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
 
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
 
    accuracy = total_correct / (len(dataloader.dataset) * NUM_CLASSES)
 
    print(
        f"Evaluation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    return precision, recall, f1, accuracy
 
 
from torch.utils.data import DataLoader, TensorDataset
 
# Define paths for testing data
test_image_path = "/Users/silviucovaci/MASTER-AI-2023/Sisteme Inteligente/constelations-guru/test_img/"
test_label_path = "/Users/silviucovaci/MASTER-AI-2023/Sisteme Inteligente/constelations-guru/__Constellation.v1i.yolov8/test/labels"
 
# Load testing data
test_images, test_labels, test_filenames = load_data(test_image_path, test_label_path)
test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
 
# Load the saved model
model = ConstellationRNN(num_classes=NUM_CLASSES, hidden_size=64)
model.load_state_dict(torch.load(r'/Users/silviucovaci/MASTER-AI-2023/Sisteme Inteligente/constelations-guru/app/models/constellation_rnn_final_1.0.pth'), weights_only=True)
model.eval()
 
# Evaluate the model on the testing data and save to CSV
evaluate_model(model, test_loader, test_filenames)