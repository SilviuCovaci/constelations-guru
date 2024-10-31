
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # pentru modelul Faster R-CNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # importul predictorului
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image

import os
import cv2
import numpy as np
import pandas as pd
import uuid

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def collate_fn(batch):
    return tuple(zip(*batch))

class ConstellationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(os.path.join(self.root_dir, annotation_file), delimiter=',', header=0)
        print(self.annotations['filename'].head())
        self.unique_images = self.annotations['filename'].unique()
        self.classes = ['background'] + sorted(self.annotations['class'].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    
    def _len_(self):
        return len(self.unique_images)
    
    def _getitem_(self, idx):
        img_name = self.unique_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        img_annotations = self.annotations[self.annotations['filename'] == img_name]
        boxes = []
        labels = []
        
        for _, row in img_annotations.iterrows():
            boxes.append([float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])])
            labels.append(self.class_to_idx[row['class']])
        
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        return image, target
    
class CustomFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CustomFasterRCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images, targets=None):
        if targets is not None:
            for t in targets:
                t['boxes'] = t['boxes'].to(images[0].device)
                t['labels'] = t['labels'].to(images[0].device)
        return self.model(images, targets)
    
constelation_cnn_model = None
constelation_cnn_transform = None

save_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + '..' + os.sep + '..'  + os.sep + '..' + os.sep + 'temp' + os.sep

class_names = {
    0: 'background',
    1: 'aquila',
    2: 'bootes',
    3: 'canis_major',
    4: 'canis_minor',
    5: 'cassiopeia',
    6: 'cygnus',
    7: 'gemini',
    8: 'leo',
    9: 'lyra',
    10: 'moon',
    11: 'orion',
    12: 'pleiades',
    13: 'sagittarius',
    14: 'scorpius',
    15: 'taurus',
    16: 'ursa_major'
}

def load_model(device):
    global class_names;
    current_folder = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = current_folder + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'constellation_model_epoch_10.pth'
    
    num_classes = len(class_names)
    model = CustomFasterRCNN(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform

def predict_image(image, confidence_threshold=0.5):
    global constelation_cnn_model, constelation_cnn_transform, save_dir, class_names;
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if constelation_cnn_model is None:
        constelation_cnn_model, constelation_cnn_transform = load_model(device)
        
    image_tensor = constelation_cnn_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = constelation_cnn_model(image_tensor)
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    label_names = [class_names[label] for label in labels]
    
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    label_names = [class_names[label] for label in labels]
    
    output_img_path = save_dir + 'cnn' + os.sep + f"{uuid.uuid4()}.jpg"
    detected_constellations, output_img_path = get_constelations_and_draw_boxes_and_labels(image, boxes, scores, labels, label_names, output_img_path)
    
    
    return detected_constellations, output_img_path;    

# Funcția pentru a desena bounding box-uri și etichete
def get_constelations_and_draw_boxes_and_labels(image, boxes, scores, labels, labels_names, output_img_path):    # Obține dimensiunile imaginii originale
    image_height, image_width = image.shape[:2]

    dpi = 100
    figsize = (image_width / dpi, image_height / dpi)

    plt.figure(figsize=figsize, dpi=dpi)
    
    #plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    detected_constellations = []
    for box, score, class_id, label_name in zip(boxes, scores, labels, labels_names):
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            box[0], box[1] - 5,
            f'{label_name}: {score:.2f}',
            color='red',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        # Adaugă informațiile despre constelații detectate
        detected_constellations.append({
            "name": label_name,  # Folosește numele constelației
            "class_id": class_id,
            "confidence": float(score),
            "bounding_box": box.tolist()  # Convertim în listă pentru a o putea trimite la template
            
        })

    plt.axis('off')
    plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()  # Închide figura pentru a elibera memorie
    
        
    detected_constellations = np.array([
        (item["name"], item["class_id"], item["confidence"], item["bounding_box"])
        for item in detected_constellations
    ], dtype=[("name", "U20"), ("class_id", "i4"), ("confidence", "f4"), ("bounding_box", "4i4")])

    return detected_constellations, output_img_path;

if __name__=='__main__': # a simple test of this class

    img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/../../../test_img/star-2630050_1280.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_constellations, output_img = predict_image(img)
    
    if (len(detected_constellations)>0):
        print("Constelație detectată!", detected_constellations);
        # Transformă lista de dicționare într-un DataFrame
        df = pd.DataFrame(detected_constellations[['name', 'confidence']])
        detected_constellations = df.groupby('name').aggregate({'confidence': 'max'}).sort_values(by='confidence').reset_index()
        print(detected_constellations.head())
        print("JSON:", detected_constellations)

    else:
        print("Nu a fost detectată nicio constelație.")