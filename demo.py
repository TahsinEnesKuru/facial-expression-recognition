import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from utils.hog import HOGDescriptor
from utils.sift import SIFT
import joblib

class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

def predict_with_custom_cnn(img):
    # Transform setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and prepare the model
    model_path = 'models/cnn.pth'  # Update with the path to your model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FacialExpressionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process the uploaded file
    image = transform(img).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predicted_expression = labels[preds.cpu().numpy()[0]]
    return predicted_expression

# Function to load ML models
def load_ml_models(model_name):
    model_paths = {
        'KNN': 'models/knn_hog_expression_classifier_ds2.pkl',
        'SVM': 'models/svm_hog_expression_classifier_ds2.pkl',
        'RF': 'models/random_forest_hog_expression_classifier_ds2.pkl',
        'MLP': 'models/mlp_hog_expression_classifier_ds2.pkl',
    }
    model = joblib.load(model_paths[model_name])
    return model

# Streamlit UI
st.title('Facial Expression Classifier')

model_options = ['SIFT Feature Matching', 'KNN', 'SVM', 'RF', 'MLP', 'CNN VGG19', 'CNN RESNET', 'CNN']
selected_model = st.selectbox('Select a model for classification:', model_options)

# Initialize HOG Descriptor
hog_descriptor = HOGDescriptor()

# Image upload
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Function to compute and cache SIFT features
def compute_sift_features():
    if 'reference_features' not in st.session_state:
        train_folder = 'datasets/dataset1/train'
        sift = SIFT()
        reference_features = {}
        for label in os.listdir(train_folder):
            if label not in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
                continue
            path = os.path.join(train_folder, label)
            st.write(f'Starting feature extraction for {label}')
            reference_images = []
            for filename in os.listdir(path):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    reference_images.append(img)
            reference_features[label] = sift.extract_sift_features(reference_images)
        st.session_state.reference_features = reference_features

def predict_with_cnn_vgg19(image):
    # Transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the model
    model_path = 'models/cnn_vgg.pth'  # Make sure this path is correct
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.vgg19(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Transform the uploaded image for model input
    image = transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predicted_expression = labels[preds.cpu().numpy()[0]]
    return predicted_expression

def predict_with_cnn_resnet50(img):
   
    # Transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the model
    model_path = 'models/cnn_resnet.pth'  # Make sure this path is correct
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Transform the uploaded image for model input
    image = transform(img).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predicted_expression = labels[preds.cpu().numpy()[0]]
    return predicted_expression

if selected_model == 'SIFT Feature Matching' and not st.session_state.get('reference_features'):
    compute_sift_features()

if image_file is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    cnn_img = Image.open(image_file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    resized_img = cv2.resize(img, (48, 48))

    if st.button('Classify Expression'):
        if selected_model in ['KNN', 'SVM', 'RF', 'MLP']:  # Machine Learning Models
            model = load_ml_models(selected_model)
            if model is not None:
                hog_feature = hog_descriptor.compute(resized_img)
                labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
                prediction = model.predict([hog_feature.flatten()])
                st.write(f'Predicted Expression: {labels[prediction[0]]}')
            else:
                st.write("Error: Model not loaded correctly.")
        elif selected_model == 'SIFT Feature Matching':  # SIFT Feature Matching
            if 'reference_features' in st.session_state:
                sift = SIFT()
                predicted_expression, all_matches = sift.classify_expression(resized_img, st.session_state.reference_features)
                st.write(f'The predicted expression is: {predicted_expression}')
            else:
                st.write("Error: SIFT features not computed.")
        elif selected_model == 'CNN VGG19':
            predicted_expression = predict_with_cnn_vgg19(cnn_img)
            st.write(f'Predicted Expression: {predicted_expression}')
        elif selected_model == 'CNN RESNET':
            predicted_expression = predict_with_cnn_resnet50(cnn_img)
            st.write(f'Predicted Expression: {predicted_expression}')  
        else:
            predicted_expression = predict_with_custom_cnn(cnn_img)
            st.write(f'Predicted Expression: {predicted_expression}')  
